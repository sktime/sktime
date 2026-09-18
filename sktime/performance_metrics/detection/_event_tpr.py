"""Event true positive rate, for alarms raised ahead of true events."""

import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric
from sktime.performance_metrics.detection.utils._match import (
    _match_alarms_to_events,
    _refuse_intervals,
)

__author__ = ["yash-sangwan"]
__all__ = ["EventTPR"]


class EventTPR(BaseDetectionMetric):
    """Event true positive rate, share of true events hit by an alarm.

    A true event at time ``T`` counts as hit if at least one alarm falls in the
    window ``[T - max_lead, T + max_delay]``. The score is the number of hit
    events, divided by the number of true events.

    Positions in ``y_true`` and ``y_pred`` are ``iloc`` references into ``X``,
    and are mapped through ``X.index`` before matching, so ``X`` is required.
    If ``X`` has a time index, ``max_lead`` and ``max_delay`` are time offsets.
    Otherwise they are in the units of ``X.index``.

    One alarm may hit more than one true event, if the event windows overlap.

    If there are no true events, the score is not defined, and ``nan`` is
    returned. If there are true events but no alarms, the score is 0.

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

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import EventTPR
    >>> X = pd.DataFrame({"foo": range(10)})
    >>> y_true = pd.DataFrame({"ilocs": [4, 8]})
    >>> y_pred = pd.DataFrame({"ilocs": [3]})
    >>> metric = EventTPR(max_lead=2)
    >>> metric(y_true, y_pred, X)
    0.5
    """

    _tags = {
        "scitype:y": "points",
        "requires_X": True,  # event positions are mapped through X.index
        "requires_y_true": True,
        "lower_is_better": False,  # higher share of hit events is better
    }

    def __init__(self, max_lead=0, max_delay=0):
        self.max_lead = max_lead
        self.max_delay = max_delay

        super().__init__()

    def _coerce_to_detection_type(self, y, X, allow_none=False):
        """Refuse interval events, then coerce as in the base class.

        The base class would turn interval events into their end points without
        notice. This metric scores point events only, so it raises instead.
        """
        _refuse_intervals(y, type(self).__name__)
        return super()._coerce_to_detection_type(y, X, allow_none=allow_none)

    def _evaluate(self, y_true, y_pred, X):
        """Evaluate the event true positive rate on given inputs.

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
            Share of true events with at least one hit,
            or ``nan`` if there are no true events.
        """
        match = _match_alarms_to_events(
            y_true, y_pred, X, max_lead=self.max_lead, max_delay=self.max_delay
        )

        if len(match.hit) == 0:
            return np.nan

        return float(np.mean(match.hit))

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
        param2 = {"max_lead": 3, "max_delay": 1}

        return [param0, param1, param2]
