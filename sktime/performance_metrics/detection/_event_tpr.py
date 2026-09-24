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
    window ``[T + min_offset, T + max_offset]``. The score is the number
    of hit events, divided by the number of true events.

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

    Positions in ``y_true`` and ``y_pred`` are ``iloc`` references into ``X``,
    and are mapped through ``X.index`` before matching, so ``X`` is required.
    If ``X`` has a time index, the offsets are time offsets, for instance
    ``pd.Timedelta("-3s")``. Otherwise they are in the units of ``X.index``.

    One alarm may hit more than one true event, if the event windows overlap.

    Only point events are scored, so interval ``ilocs`` (segments) in
    ``y_true`` or ``y_pred`` raise a ``ValueError``.

    If there are no true events, the score is not defined, and ``nan`` is
    returned. If there are true events but no alarms, the score is 0.

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

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import EventTPR
    >>> X = pd.DataFrame({"foo": range(10)})
    >>> y_true = pd.DataFrame({"ilocs": [4, 8]})
    >>> y_pred = pd.DataFrame({"ilocs": [3]})
    >>> metric = EventTPR(min_offset=-2)
    >>> metric(y_true, y_pred, X)
    0.5
    """

    _tags = {
        "scitype:y": "points",
        "requires_X": True,  # event positions are mapped through X.index
        "requires_y_true": True,
        "lower_is_better": False,  # higher share of hit events is better
    }

    def __init__(self, min_offset=0, max_offset=0):
        self.min_offset = min_offset
        self.max_offset = max_offset

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
            y_true,
            y_pred,
            X,
            min_offset=self.min_offset,
            max_offset=self.max_offset,
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
        param1 = {"min_offset": -2}
        param2 = {"min_offset": -3, "max_offset": 1}

        return [param0, param1, param2]
