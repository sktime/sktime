"""Matching of detected alarms to true events, for live detection metrics."""

import datetime as dt
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd

__author__ = ["yash-sangwan"]


@dataclass
class _EventMatch:
    """Result of matching alarms to true events.

    Attributes
    ----------
    event_times : pd.Index, length n_events
        Times of the true events, taken from ``X.index``.
        Same order as the rows of ``y_true``.
    alarm_times : pd.Index, length n_alarms
        Times of the alarms, taken from ``X.index``.
        Same order as the rows of ``y_pred``.
    hit : 1D np.ndarray of bool, length n_events
        Whether the event at the same position has at least one alarm
        in its window.
    earliest_hit : 1D np.ndarray of int, length n_events
        Position, in ``y_pred``, of the earliest alarm in the event window.
        Entries are ``-1`` where the event was not hit.
    false_alarm : 1D np.ndarray of bool, length n_alarms
        Whether the alarm at the same position falls in no event window.
    """

    event_times: pd.Index
    alarm_times: pd.Index
    hit: np.ndarray
    earliest_hit: np.ndarray
    false_alarm: np.ndarray


def _event_times(events, X, var_name):
    """Map the ``ilocs`` column of an event table onto the index of ``X``.

    Parameters
    ----------
    events : pd.DataFrame
        Event table in points format, with an ``"ilocs"`` column.
    X : pd.DataFrame or pd.Series
        Time series the events refer to. Only its index is used.
    var_name : str
        Name of ``events`` in the caller, used in error messages.

    Returns
    -------
    pd.Index
        Times of the events, in the order of the rows of ``events``.

    Raises
    ------
    ValueError
        If ``events`` has no ``"ilocs"`` column, is not in points format,
        or has positions outside ``[0, len(X))``.
    """
    if "ilocs" not in events.columns:
        raise ValueError(
            f"{var_name} must have an 'ilocs' column, but found columns "
            f"{list(events.columns)}."
        )

    ilocs = events["ilocs"]
    if len(ilocs) > 0 and not pd.api.types.is_integer_dtype(ilocs.dtype):
        raise ValueError(
            f"{var_name} must be in points format, with integer 'ilocs', "
            f"but found dtype {ilocs.dtype}. Segments are not supported."
        )

    positions = np.asarray(ilocs.to_numpy(), dtype=int)

    # a negative position would silently wrap to the end of X, so refuse it
    n_timepoints = len(X.index)
    outside = (positions < 0) | (positions >= n_timepoints)
    if outside.any():
        raise ValueError(
            f"{var_name} has 'ilocs' outside [0, {n_timepoints}), the positions "
            f"of X, found {positions[outside].tolist()}. "
            "'ilocs' must be positions into the X passed with them."
        )

    return X.index[positions]


def _is_time_index(index):
    """Return whether ``index`` is a time index, with time offsets as units.

    Only a ``pd.DatetimeIndex`` counts as a time index. Any other index is used
    in its own units, the values of the index.

    Parameters
    ----------
    index : pd.Index
        Index of the time series.

    Returns
    -------
    bool
        True if ``index`` is a ``pd.DatetimeIndex``, False otherwise.
    """
    return isinstance(index, pd.DatetimeIndex)


def _refuse_intervals(y, metric_name):
    """Raise if an event table holds intervals, as live metrics score points only.

    Without this check, the detection metric base class turns interval events
    into their end points without notice, which changes the events that are
    scored. Empty tables pass, since an empty alarm table is valid.

    Parameters
    ----------
    y : pd.DataFrame or None
        Event table to check, with an ``"ilocs"`` column.
    metric_name : str
        Name of the calling metric, used in the error message.

    Raises
    ------
    ValueError
        If ``y`` is a non-empty table whose ``"ilocs"`` are intervals.
    """
    if y is None or len(y) == 0 or "ilocs" not in y.columns:
        return

    if isinstance(y["ilocs"].dtype, pd.IntervalDtype):
        raise ValueError(
            f"{metric_name} scores point events only, but found interval "
            "'ilocs', that is segments. Pass one row per event, with integer "
            "'ilocs'. Intervals are refused, as turning them into end points "
            "would change the events."
        )


def _coerce_offset(value, index, var_name):
    """Coerce a signed window offset to the unit of ``index``.

    Parameters
    ----------
    value : int, float, or time offset
        Offset to coerce. Negative values are before the event, positive
        values are after it.
    index : pd.Index
        Index of the time series the offset applies to.
    var_name : str
        Name of ``value`` in the caller, used in error messages.

    Returns
    -------
    pd.Timedelta, if ``index`` is a time index, otherwise ``value`` unchanged.

    Raises
    ------
    TypeError
        If the unit of ``value`` does not fit the unit of ``index``.
        A plain ``0`` is accepted for both, and means no offset.
    ValueError
        If ``value`` is NaN.
    """
    is_time_index = _is_time_index(index)
    is_time_value = isinstance(value, (pd.Timedelta, np.timedelta64, dt.timedelta))
    is_zero = isinstance(value, (int, np.integer)) and value == 0

    if is_time_index:
        if is_zero:
            return pd.Timedelta(0)
        if not is_time_value and not isinstance(value, str):
            raise TypeError(
                f"{var_name} must be a time offset, for instance "
                f"pd.Timedelta('-3s'), if X has a time index, "
                f"but found {value!r}."
            )
        offset = pd.Timedelta(value)
    else:
        if is_time_value or not isinstance(value, numbers.Real):
            raise TypeError(
                f"{var_name} must be a number in the units of X.index, "
                f"if X does not have a time index, but found {value!r}."
            )
        offset = value

    # NaN would bend the window without notice, as every comparison with it fails
    if pd.isna(offset):
        raise ValueError(f"{var_name} must not be NaN, but found {value!r}.")

    return offset


def _match_alarms_to_events(y_true, y_pred, X, min_offset=0, max_offset=0):
    """Match detected alarms to true events, on the time axis of ``X``.

    An alarm hits a true event at time ``T`` if it falls in the closed window
    ``[T + min_offset, T + max_offset]``. The offsets are signed,
    negative is before the event and positive is after it.

    Positions in ``y_true`` and ``y_pred`` are ``iloc`` references into ``X``,
    and are mapped through ``X.index`` before matching. If ``X`` has a time
    index, windows and returned times are in time units. Otherwise they are
    in the units of ``X.index``, not in positions: on an index ``[0, 10, 20]``,
    an ``min_offset`` of -10 reaches back one point, not ten.

    An alarm may hit more than one event, if event windows overlap. Alarms
    that hit no event are false alarms. Further alarms inside a window that
    is already hit are not false alarms.

    Parameters
    ----------
    y_true : pd.DataFrame
        True events, in points format, with an ``"ilocs"`` column.
    y_pred : pd.DataFrame
        Detected alarms, in points format, with an ``"ilocs"`` column.
    X : pd.DataFrame or pd.Series
        Time series the events refer to. Only its index is used.
    min_offset : int, float, or time offset, default=0
        Start of the hit window, relative to the event time ``T``.
        A number in the units of ``X.index``, or a time offset such as
        ``pd.Timedelta("-3s")`` if ``X`` has a time index.
    max_offset : int, float, or time offset, default=0
        End of the hit window, relative to the event time ``T``.
        Same unit as ``min_offset``. Must not be before ``min_offset``.

    Returns
    -------
    _EventMatch
        Which events were hit, the earliest hit per event, and which alarms
        hit no event. See the class docstring for the fields.

    Raises
    ------
    ValueError
        If an ``"ilocs"`` value is outside ``[0, len(X))``, if an offset is NaN,
        or if ``min_offset`` is after ``max_offset``.
    TypeError
        If the unit of an offset does not fit ``X.index``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection.utils._match import (
    ...     _match_alarms_to_events
    ... )
    >>> X = pd.DataFrame({"foo": range(10)})
    >>> y_true = pd.DataFrame({"ilocs": [5]})
    >>> y_pred = pd.DataFrame({"ilocs": [3, 8]})
    >>> match = _match_alarms_to_events(y_true, y_pred, X, min_offset=-2)
    >>> match.hit
    array([ True])
    >>> match.earliest_hit
    array([0])
    >>> match.false_alarm
    array([False,  True])
    """
    event_times = _event_times(y_true, X, "y_true")
    alarm_times = _event_times(y_pred, X, "y_pred")

    min_off = _coerce_offset(min_offset, X.index, "min_offset")
    max_off = _coerce_offset(max_offset, X.index, "max_offset")

    # an inverted window would silently match nothing, so refuse it
    if min_off > max_off:
        raise ValueError(
            "min_offset must not be after max_offset, but found "
            f"min_offset={min_offset!r} and "
            f"max_offset={max_offset!r}. "
            "The hit window is [T + min_offset, T + max_offset]."
        )

    n_events = len(event_times)
    n_alarms = len(alarm_times)

    hit = np.zeros(n_events, dtype=bool)
    earliest_hit = np.full(n_events, -1, dtype=int)
    false_alarm = np.ones(n_alarms, dtype=bool)

    if n_events == 0 or n_alarms == 0:
        return _EventMatch(
            event_times=event_times,
            alarm_times=alarm_times,
            hit=hit,
            earliest_hit=earliest_hit,
            false_alarm=false_alarm,
        )

    # sort alarms, so that the first alarm in a window is the earliest one
    order = np.argsort(alarm_times.to_numpy(), kind="stable")
    sorted_alarms = alarm_times[order]

    # half-open positions of the window bounds, in the sorted alarms
    start = sorted_alarms.searchsorted(event_times + min_off, side="left")
    stop = sorted_alarms.searchsorted(event_times + max_off, side="right")

    hit = stop > start
    earliest_hit[hit] = order[start[hit]]

    for window_start, window_stop in zip(start[hit], stop[hit]):
        false_alarm[order[window_start:window_stop]] = False

    return _EventMatch(
        event_times=event_times,
        alarm_times=alarm_times,
        hit=hit,
        earliest_hit=earliest_hit,
        false_alarm=false_alarm,
    )
