"""Mean detection offset, signed distance from each event to its earliest alarm."""

import numpy as np
import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric
from sktime.performance_metrics.detection.utils._match import (
    _is_time_index,
    _match_alarms_to_events,
    _refuse_intervals,
)

__author__ = ["yash-sangwan"]
__all__ = ["MeanDetectionOffset"]


class MeanDetectionOffset(BaseDetectionMetric):
    """Mean detection offset, how far the earliest alarm is from each event.

    A true event at time ``T`` counts as hit if at least one alarm falls in the
    window ``[T + min_offset, T + max_offset]``. For each hit event, the
    detection offset is the time of the earliest alarm in its window, minus
    ``T``. The score is the mean detection offset over hit events only.

    The offset is signed in the same way as the window: it is negative for an
    alarm before the event, and positive for a late hit, which can only happen
    if ``max_offset`` is above 0. An earlier alarm gives a smaller score, so
    lower is better.

    The offsets are signed, negative is before the event and positive is after
    it. With offsets in the units of ``X.index``, for an event at ``T``:

    * ``min_offset=0, max_offset=0``: only an alarm exactly at ``T``.
    * ``min_offset=-3, max_offset=0``: advance only, an alarm from 3
      before ``T`` up to ``T``. Late alarms do not count.
    * ``min_offset=0, max_offset=2``: late only, an alarm from ``T`` up
      to 2 after ``T``. Early alarms do not count.
    * ``min_offset=-3, max_offset=2``: before and after, an alarm from 3
      before ``T`` up to 2 after ``T``.
    * ``min_offset=-10, max_offset=-2``: at least 2 before ``T``, and not
      earlier than 10 before ``T``. An alarm at ``T`` does not count.

    Missed events do not enter the mean. Read this score together with
    ``EventTPR``, which reports how many events were hit at all.

    Positions in ``y_true`` and ``y_pred`` are ``iloc`` references into ``X``,
    and are mapped through ``X.index`` before matching, so ``X`` is required.
    If ``X`` has a time index, the offsets are time offsets, for instance
    ``pd.Timedelta("-3s")``, and the score is returned as a number of
    ``time_unit``. Otherwise all values are in the units of ``X.index``.

    Only point events are scored, so interval ``ilocs`` (segments) in
    ``y_true`` or ``y_pred`` raise a ``ValueError``.

    If there are no true events, or no event is hit, the score is not defined,
    and ``nan`` is returned.

    Parameters
    ----------
    min_offset : int, float, or time offset, default=0
        Start of the hit window, relative to the event time ``T``.
        Negative values let alarms before the event count.
        A time offset, for instance ``pd.Timedelta("-3s")``, if ``X`` has a
        time index, otherwise a number in the units of ``X.index``.
        A ``ValueError`` is raised if it is NaN, or after ``max_offset``.
    max_offset : int, float, or time offset, default=0
        End of the hit window, relative to the event time ``T``.
        Positive values let alarms after the event count, the default of 0
        means that alarms after the event do not count.
        Same unit as ``min_offset``. NaN raises a ``ValueError``.
    time_unit : str, default="s"
        Unit of the returned score, if ``X`` has a time index.
        Any unit accepted by ``pd.Timedelta``, for instance ``"s"``, ``"ms"``,
        ``"min"``, or ``"h"``. Ignored if ``X`` does not have a time index.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import MeanDetectionOffset
    >>> index = pd.date_range("2020-01-01", periods=20, freq="s")
    >>> X = pd.DataFrame({"foo": range(20)}, index=index)
    >>> y_true = pd.DataFrame({"ilocs": [5, 15]})
    >>> y_pred = pd.DataFrame({"ilocs": [2, 14]})
    >>> metric = MeanDetectionOffset(min_offset=pd.Timedelta("-3s"))
    >>> metric(y_true, y_pred, X)
    -2.0
    """

    _tags = {
        "scitype:y": "points",
        "requires_X": True,  # event positions are mapped through X.index
        "requires_y_true": True,
        "lower_is_better": True,  # an earlier alarm gives a smaller offset
    }

    def __init__(self, min_offset=0, max_offset=0, time_unit="s"):
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.time_unit = time_unit

        super().__init__()

    def _coerce_to_detection_type(self, y, X, allow_none=False):
        """Refuse interval events, then coerce as in the base class.

        The base class would turn interval events into their end points without
        notice. This metric scores point events only, so it raises instead.
        """
        _refuse_intervals(y, type(self).__name__)
        return super()._coerce_to_detection_type(y, X, allow_none=allow_none)

    def _evaluate(self, y_true, y_pred, X):
        """Evaluate the mean detection offset on given inputs.

        private _evaluate containing core logic, called from evaluate

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth events, in points format, with an ``"ilocs"`` column.
        y_pred : pd.DataFrame
            Detected alarms, in points format, with an ``"ilocs"`` column.
        X : pd.DataFrame
            Time series the events refer to. Only its index is used.

        Returns
        -------
        float
            Mean of earliest hit time minus event time, over hit events only,
            or ``nan`` if no event is hit.
        """
        match = _match_alarms_to_events(
            y_true,
            y_pred,
            X,
            min_offset=self.min_offset,
            max_offset=self.max_offset,
        )

        if not match.hit.any():
            return np.nan

        hit_times = match.event_times[match.hit]
        earliest_alarm_times = match.alarm_times[match.earliest_hit[match.hit]]
        offset = earliest_alarm_times - hit_times

        # a time index gives a TimedeltaIndex, which has a mean in time units,
        # a plain numeric Index has no mean method, so go through numpy
        if _is_time_index(X.index):
            return float(offset.mean() / pd.Timedelta(1, unit=self.time_unit))
        return float(np.mean(offset.to_numpy()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        param0 = {}
        param1 = {"min_offset": -2}
        param2 = {"min_offset": -3, "max_offset": 1, "time_unit": "ms"}

        return [param0, param1, param2]
