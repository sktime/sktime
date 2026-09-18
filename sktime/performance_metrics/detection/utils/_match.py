"""Matching of detected alarms to true events, for live detection metrics."""

import datetime as dt
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

    return X.index[np.asarray(ilocs.to_numpy(), dtype=int)]


def _coerce_tolerance(value, index, var_name):
    """Coerce a tolerance to the unit of ``index``.

    Parameters
    ----------
    value : int, float, or time offset
        Tolerance to coerce.
    index : pd.Index
        Index of the time series the tolerance applies to.
    var_name : str
        Name of ``value`` in the caller, used in error messages.

    Returns
    -------
    pd.Timedelta, if ``index`` is a time index, otherwise ``value`` unchanged.

    Raises
    ------
    TypeError
        If the unit of ``value`` does not fit the unit of ``index``.
        A plain ``0`` is accepted for both, and means no tolerance.
    """
    is_time_index = isinstance(index, pd.DatetimeIndex)
    is_time_value = isinstance(value, (pd.Timedelta, np.timedelta64, dt.timedelta))
    is_zero = isinstance(value, (int, np.integer)) and value == 0

    if is_time_index:
        if is_zero:
            return pd.Timedelta(0)
        if not is_time_value and not isinstance(value, str):
            raise TypeError(
                f"{var_name} must be a time offset, for instance "
                f"pd.Timedelta('3s'), if X has a time index, "
                f"but found {value!r}."
            )
        return pd.Timedelta(value)

    if is_time_value:
        raise TypeError(
            f"{var_name} must be a number of index steps, if X does not have "
            f"a time index, but found {value!r}."
        )
    return value


def _match_alarms_to_events(y_true, y_pred, X, max_lead, max_delay=0):
    """Match detected alarms to true events, on the time axis of ``X``.

    An alarm hits a true event at time ``T`` if it falls in the closed window
    ``[T - max_lead, T + max_delay]``.

    Positions in ``y_true`` and ``y_pred`` are ``iloc`` references into ``X``,
    and are mapped through ``X.index`` before matching. If ``X`` has a time
    index, windows and returned times are in time units. Otherwise they are
    in index steps.

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
    max_lead : int, or time offset
        How early an alarm may be, and still count as a hit.
        Number of index steps, or a time offset such as ``pd.Timedelta("3s")``
        if ``X`` has a time index.
    max_delay : int, or time offset, default=0
        How late an alarm may be, and still count as a hit.
        Same unit as ``max_lead``. The default means that an alarm after the
        event does not count.

    Returns
    -------
    _EventMatch
        Which events were hit, the earliest hit per event, and which alarms
        hit no event. See the class docstring for the fields.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection.utils._match import (
    ...     _match_alarms_to_events
    ... )
    >>> X = pd.DataFrame({"foo": range(10)})
    >>> y_true = pd.DataFrame({"ilocs": [5]})
    >>> y_pred = pd.DataFrame({"ilocs": [3, 8]})
    >>> match = _match_alarms_to_events(y_true, y_pred, X, max_lead=2)
    >>> match.hit
    array([ True])
    >>> match.earliest_hit
    array([0])
    >>> match.false_alarm
    array([False,  True])
    """
    event_times = _event_times(y_true, X, "y_true")
    alarm_times = _event_times(y_pred, X, "y_pred")

    lead = _coerce_tolerance(max_lead, X.index, "max_lead")
    delay = _coerce_tolerance(max_delay, X.index, "max_delay")

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
    start = sorted_alarms.searchsorted(event_times - lead, side="left")
    stop = sorted_alarms.searchsorted(event_times + delay, side="right")

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
