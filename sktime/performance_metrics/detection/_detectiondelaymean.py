"""Mean detection delay metric for time series event detection tasks."""

import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class DetectionDelayMean(BaseDetectionMetric):
    r"""Mean detection delay for event detection tasks.

    Computes the average delay between true events and matched predictions
    using greedy earliest-available matching.

    For each true event `t`, the metric searches for the earliest unused
    prediction in the window ``[t - early_tolerance, t + max_delay]``.
    Predictions within ``early_tolerance`` steps **before** the true event
    are considered timely and get ``delay = 0``.

    Late detections keep their positive delay. Unmatched events receive
    penalty equal to ``max_delay`` (default 1000.0).

    Lower values are better (faster average detection).

    Parameters
    ----------
    early_tolerance : int, default=0
        Number of steps a prediction can precede a true event and still
        be considered timely (delay set to 0).
    max_delay : int or None, default=None
        Caps delay and serves as penalty for unmatched or very late events.
        Tune this value based on your data's sampling frequency.

    Notes
    -----
    Input must be :class:`pandas.DataFrame` with a column named ``"ilocs"``
    containing integer event locations.
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": True,
    }

    def __init__(self, early_tolerance=0, max_delay=None):
        super().__init__()
        self.early_tolerance = early_tolerance
        self.max_delay = max_delay

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute mean detection delay.

        Parameters
        ----------
        y_true : pd.DataFrame with column "ilocs"
            True event locations.
        y_pred : pd.DataFrame with column "ilocs"
            Predicted event locations.
        X : ignored

        Returns
        -------
        float
            Mean detection delay across true events.
        """
        if len(y_true) == 0:
            return 0.0

        true_ilocs = np.sort(y_true["ilocs"].values)

        if len(y_pred) == 0:
            penalty = self.max_delay if self.max_delay is not None else 1000.0
            return float(penalty)

        pred_ilocs = np.sort(y_pred["ilocs"].values)
        used = np.zeros(len(pred_ilocs), dtype=bool)

        delays = []

        for t in true_ilocs:
            lower = t - self.early_tolerance
            upper = t + (self.max_delay if self.max_delay is not None else 10000)

            candidate_idx = np.where(
                (pred_ilocs >= lower) & (pred_ilocs <= upper) & (~used)
            )[0]

            if len(candidate_idx) == 0:
                delay = self.max_delay if self.max_delay is not None else 1000.0
            else:
                idx = candidate_idx[0]
                pred_t = pred_ilocs[idx]
                raw_delay = pred_t - t
                used[idx] = True

                # Early within tolerance → delay = 0, late keeps positive delay
                if raw_delay >= -self.early_tolerance:
                    delay = 0 if raw_delay <= 0 else raw_delay
                else:
                    delay = raw_delay

                if self.max_delay is not None:
                    delay = min(delay, self.max_delay)

            delays.append(delay)

        return float(np.mean(delays))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {},
            {"early_tolerance": 5, "max_delay": 100},
        ]
