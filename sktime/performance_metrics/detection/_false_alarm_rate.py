"""False alarm rate, alarms that hit no true event, per unit of scored time."""

import numpy as np
import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric
from sktime.performance_metrics.detection.utils._match import (
    _is_time_index,
    _match_alarms_to_events,
    _refuse_intervals,
)

__author__ = ["yash-sangwan"]
__all__ = ["FalseAlarmRate"]


class FalseAlarmRate(BaseDetectionMetric):
    """False alarm rate, number of alarms that hit no event, per unit of time.

    A true event at time ``T`` counts as hit by an alarm that falls in the
    window ``[T + min_offset, T + max_offset]``. An alarm that falls in
    no event window is a false alarm. Further alarms inside a window that is
    already hit are not false alarms.

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

    The score is the number of false alarms, divided by the scored duration.
    The scored duration is the span of ``X.index``, last value minus first
    value, so ``X`` should hold only the part of the stream that is scored.

    This metric counts unmatched alarms only. With ``min_offset=0`` and
    ``max_offset=0``, the window is the event time itself, so an alarm is
    unmatched unless it lands exactly on an event. The default is therefore
    close to the number of alarms divided by the length of ``X``, but an alarm
    that lands exactly on an event is not a false alarm.

    Positions in ``y_true`` and ``y_pred`` are ``iloc`` references into ``X``,
    and are mapped through ``X.index`` before matching, so ``X`` is required.
    If ``X`` has a time index, the offsets are time offsets, for instance
    ``pd.Timedelta("-3s")``, and the duration is counted in ``time_unit``,
    by default hours. Otherwise all values are in the units of ``X.index``,
    and ``time_unit`` is ignored.

    With no true events the score is still defined, as every alarm is a false
    alarm. With no alarms the score is 0, as long as ``X`` has a span. If ``X``
    has no span, that is fewer than two time points, the rate is not defined,
    and ``nan`` is returned, even if there are no alarms.

    Only point events are scored, so interval ``ilocs`` (segments) in
    ``y_true`` or ``y_pred`` raise a ``ValueError``.

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
        Positive values let alarms after the event count, with the default of
        0, alarms after the event are false alarms.
        Same unit as ``min_offset``. NaN raises a ``ValueError``.
    time_unit : str, default="hour"
        Unit in which the duration is counted, if ``X`` has a time index,
        so the score is false alarms per ``time_unit``.
        Any unit accepted by ``pd.Timedelta``, for instance ``"hour"``,
        ``"min"``, or ``"s"``. Ignored if ``X`` does not have a time index.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import FalseAlarmRate
    >>> index = pd.date_range("2020-01-01", periods=7, freq="20min")
    >>> X = pd.DataFrame({"foo": range(7)}, index=index)
    >>> y_true = pd.DataFrame({"ilocs": [5]})
    >>> y_pred = pd.DataFrame({"ilocs": [1, 4]})
    >>> metric = FalseAlarmRate(min_offset=pd.Timedelta("-20min"))
    >>> metric(y_true, y_pred, X)
    0.5
    """

    _tags = {
        "scitype:y": "points",
        "requires_X": True,  # event positions and duration come from X.index
        "requires_y_true": True,
        "lower_is_better": True,  # fewer false alarms per unit time is better
    }

    def __init__(self, min_offset=0, max_offset=0, time_unit="hour"):
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
        """Evaluate the false alarm rate on given inputs.

        private _evaluate containing core logic, called from evaluate

        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth events, in points format, with an ``"ilocs"`` column.
        y_pred : pd.DataFrame
            Detected alarms, in points format, with an ``"ilocs"`` column.
        X : pd.DataFrame
            Time series the events refer to. Its index gives event times,
            and the scored duration.

        Returns
        -------
        float
            Number of alarms that hit no true event, divided by the span of
            ``X.index``, or ``nan`` if ``X`` has no span.
        """
        match = _match_alarms_to_events(
            y_true,
            y_pred,
            X,
            min_offset=self.min_offset,
            max_offset=self.max_offset,
        )
        n_false_alarms = int(match.false_alarm.sum())

        # the scored duration is the span of X, last index value minus first
        index = X.index
        if len(index) < 2:
            return np.nan

        duration = index[-1] - index[0]
        if _is_time_index(index):
            duration = duration / pd.Timedelta(1, unit=self.time_unit)

        # a repeated index value can still give no span, and no rate
        if duration == 0:
            return np.nan

        return float(n_false_alarms / duration)

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
        param2 = {"min_offset": -3, "max_offset": 1, "time_unit": "min"}

        return [param0, param1, param2]
