"""Advance detection score for evaluating early event detection."""

import numpy as np
import pandas as pd

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class AdvanceDetectionScore(BaseDetectionMetric):
    r"""Score for evaluating advance (early) event detection.

    Measures how well a detector identifies events *before* they occur.
    For each true event, the closest preceding detection is found and scored
    based on how far in advance it was made. Detections after the true event
    receive a penalty score (default 0).

    For true event time points :math:`B = (b_1, b_2, \ldots, b_m)` and
    detected time points :math:`A = (a_1, a_2, \ldots, a_n)`,
    the score for each true event :math:`b_j` is:

    .. math::

        s(b_j) = \begin{cases}
            1 - \frac{b_j - a^*_j}{w}
                & \text{if } 0 \leq b_j - a^*_j \leq w \\
            p & \text{if } a^*_j > b_j \text{ (late detection)} \\
            0 & \text{if no detection within window}
        \end{cases}

    where :math:`a^*_j` is the closest preceding detection to :math:`b_j`,
    :math:`w` is the maximum advance window, and :math:`p` is the late
    detection penalty.

    The total score is the sum (or mean if ``normalize=True``) of per-event
    scores.

    This scoring concept is related to time-aware detection evaluation
    as discussed by Gupta et al. (2009) [1]_ and activity advance detection
    literature.

    If ``X`` is provided, the time points are taken as the location indices in ``X``.
    Otherwise, it is assumed that ``X`` has a ``RangeIndex``.

    Parameters
    ----------
    window : int or float, default=10
        Maximum advance time window. Detections more than ``window`` time
        units before the true event receive a score of 0.
    penalty_late : float, default=0.0
        Score assigned to a true event whose closest detection is after
        the event (late detection). Must be between 0 and 1.
    normalize : bool, default=True
        If True, the total score is divided by the number of true events,
        yielding a mean per-event score in [0, 1].
        If False, the raw sum of per-event scores is returned.

    References
    ----------
    .. [1] Gupta, M., Gao, J., Aggarwal, C. C., & Han, J. (2014).
       Outlier detection for temporal data: A survey.
       IEEE Transactions on Knowledge and Data Engineering, 26(9), 2250-2267.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection._advance import (
    ...     AdvanceDetectionScore,
    ... )
    >>> y_true = pd.DataFrame({"ilocs": [10, 20]})
    >>> y_pred = pd.DataFrame({"ilocs": [5, 15]})
    >>> metric = AdvanceDetectionScore(window=10, normalize=True)
    >>> score = metric(y_true, y_pred)
    """

    _tags = {
        "scitype:y": "points",
        "requires_X": False,
        "lower_is_better": False,
    }

    def __init__(self, window=10, penalty_late=0.0, normalize=True):
        self.window = window
        self.penalty_late = penalty_late
        self.normalize = normalize

        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Evaluate the advance detection score on given inputs.

        private _evaluate containing core logic, called from evaluate

        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) event locations, in ``X``.
            Should be ``pd.DataFrame``, ``pd.Series``, or ``np.ndarray`` (1D or 2D),
            of ``Series`` scitype = individual time series.

            For further details on data format, see glossary on :term:`mtype`.

        y_pred : time series in ``sktime`` compatible data container format
            Detected events to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.

        X : optional, pd.DataFrame, pd.Series or np.ndarray
            Time series that is being labelled.
            If not provided, assumes ``RangeIndex`` for ``X``, and that
            values in ``X`` do not matter.

        Returns
        -------
        score : float
            Calculated metric.
        """
        y_true_ilocs = y_true.ilocs
        y_pred_ilocs = y_pred.ilocs

        if X is not None and not isinstance(X.index, pd.RangeIndex):
            y_true_locs = X.index[y_true_ilocs]
            y_pred_locs = X.index[y_pred_ilocs]
        else:
            y_true_locs = y_true_ilocs
            y_pred_locs = y_pred_ilocs

        y_true_locs = np.array(y_true_locs)
        y_pred_locs = np.array(y_pred_locs)

        n_true = len(y_true_locs)

        if n_true == 0:
            return 0.0

        if len(y_pred_locs) == 0:
            return 0.0

        window = self.window
        penalty_late = self.penalty_late

        total_score = 0.0

        for b in y_true_locs:
            # compute delay for each prediction: positive means advance (pred < true)
            delays = b - y_pred_locs  # positive = advance, negative = late

            # find advance detections (delay >= 0, i.e., pred is before or at event)
            advance_mask = delays >= 0
            if np.any(advance_mask):
                # among advance detections, pick the one closest to the event
                # (smallest non-negative delay)
                advance_delays = delays[advance_mask]
                best_delay = np.min(advance_delays)

                if best_delay <= window:
                    # linear decay: score = 1 at delay=0, score = 0 at delay=window
                    total_score += 1.0 - best_delay / window
                # else: detection is outside window, contributes 0
            else:
                # all detections are late (after the event)
                total_score += penalty_late

        if self.normalize:
            total_score /= n_true

        return total_score

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
        param1 = {}
        param2 = {"window": 5, "penalty_late": 0.5, "normalize": False}

        return [param1, param2]
