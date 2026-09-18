"""Mean advance time, of alarms raised ahead of true events."""

import numpy as np
import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric
from sktime.performance_metrics.detection.utils._match import (
    _is_time_index,
    _match_alarms_to_events,
    _refuse_intervals,
)

__author__ = ["yash-sangwan"]
__all__ = ["MeanAdvanceTime"]


class MeanAdvanceTime(BaseDetectionMetric):
    """Mean advance time, how early the earliest alarm comes before each event.

    A true event at time ``T`` counts as hit if at least one alarm falls in the
    window ``[T - max_lead, T + max_delay]``. For each hit event, the advance
    time is ``T`` minus the time of the earliest alarm in its window.
    The score is the mean advance time over hit events only.

    Missed events do not enter the mean. Read this score together with
    ``EventTPR``, which reports how many events were hit at all.

    Positions in ``y_true`` and ``y_pred`` are ``iloc`` references into ``X``,
    and are mapped through ``X.index`` before matching, so ``X`` is required.
    If ``X`` has a time index, ``max_lead`` and ``max_delay`` are time offsets,
    and the score is returned as a number of ``time_unit``.
    Otherwise all values are in the units of ``X.index``.

    The advance time is positive for alarms before the event. It is negative
    for a late hit, which can only happen if ``max_delay`` is above 0.

    If there are no true events, or no event is hit, the score is not defined,
    and ``nan`` is returned.

    Parameters
    ----------
    max_lead : int or time offset, default=0
        How early an alarm may be, and still hit a true event.
        A time offset, for instance ``pd.Timedelta("3s")``, if ``X`` has a
        time index, otherwise a number in the units of ``X.index``.
        The default of 0 counts only alarms at the event itself,
        or after it within ``max_delay``.
    max_delay : int or time offset, default=0
        How late an alarm may be, and still hit a true event.
        Same unit as ``max_lead``. The default of 0 means that alarms
        after the event do not count.
    time_unit : str, default="s"
        Unit of the returned score, if ``X`` has a time index.
        Any unit accepted by ``pd.Timedelta``, for instance ``"s"``, ``"ms"``,
        ``"min"``, or ``"h"``. Ignored if ``X`` does not have a time index.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import MeanAdvanceTime
    >>> index = pd.date_range("2020-01-01", periods=20, freq="s")
    >>> X = pd.DataFrame({"foo": range(20)}, index=index)
    >>> y_true = pd.DataFrame({"ilocs": [5, 15]})
    >>> y_pred = pd.DataFrame({"ilocs": [2, 14]})
    >>> metric = MeanAdvanceTime(max_lead=pd.Timedelta("3s"))
    >>> metric(y_true, y_pred, X)
    2.0
    """

    _tags = {
        "scitype:y": "points",
        "requires_X": True,  # event positions are mapped through X.index
        "requires_y_true": True,
        "lower_is_better": False,  # earlier alarms are better
    }

    def __init__(self, max_lead=0, max_delay=0, time_unit="s"):
        self.max_lead = max_lead
        self.max_delay = max_delay
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
        """Evaluate the mean advance time on given inputs.

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
            Mean of event time minus earliest hit time, over hit events only,
            or ``nan`` if no event is hit.
        """
        match = _match_alarms_to_events(
            y_true, y_pred, X, max_lead=self.max_lead, max_delay=self.max_delay
        )

        if not match.hit.any():
            return np.nan

        hit_times = match.event_times[match.hit]
        earliest_alarm_times = match.alarm_times[match.earliest_hit[match.hit]]
        advance = hit_times - earliest_alarm_times

        # a time index gives a TimedeltaIndex, which has a mean in time units,
        # a plain numeric Index has no mean method, so go through numpy
        if _is_time_index(X.index):
            return float(advance.mean() / pd.Timedelta(1, unit=self.time_unit))
        return float(np.mean(advance.to_numpy()))

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
        param1 = {"max_lead": 2}
        param2 = {"max_lead": 3, "max_delay": 1, "time_unit": "ms"}

        return [param0, param1, param2]
