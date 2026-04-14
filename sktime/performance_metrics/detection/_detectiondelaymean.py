"""Mean detection delay metric for time series event detection tasks."""

import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class DetectionDelayMean(BaseDetectionMetric):
    r"""Mean detection delay for event detection tasks.

    For each true event, finds the earliest unused prediction within
    ``early_tolerance`` before or any time after the event.
    Early detections within tolerance are clipped to delay = 0.
    Unmatched events receive penalty (``max_delay`` or 1000.0 by default).

    Uses greedy earliest-available matching. Supports multiple events/predictions.
    Lower is better.

    Parameters
    ----------
    early_tolerance : int, default=0
        Steps a prediction can precede a true event and still be timely (delay=0).
    max_delay : int or None, default=None
        Caps delay and serves as penalty for unmatched events.
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": True,
    }

    def __init__(self, early_tolerance=0, max_delay=None):
        super().__init__()  # MUST be first in sktime base classes
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
            Mean detection delay across all true events.
        """
        if len(y_true) == 0:
            return 0.0

        true_ilocs = np.sort(y_true["ilocs"].values)
        pred_ilocs = (
            np.sort(y_pred["ilocs"].values) if len(y_pred) > 0 else np.array([])
        )

        delays = []
        used = np.zeros(len(pred_ilocs), dtype=bool)

        for t in true_ilocs:
            candidates = np.where((pred_ilocs >= t - self.early_tolerance) & (~used))[0]

            if len(candidates) == 0:
                delay = self.max_delay if self.max_delay is not None else 1000.0
            else:
                idx = candidates[0]
                delay = pred_ilocs[idx] - t
                used[idx] = True
                delay = max(0, delay)
                if self.max_delay is not None:
                    delay = min(delay, self.max_delay)

            delays.append(delay)

        return float(np.mean(delays))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.

        Returns
        -------
        list of dict
            List of parameter dicts to test.
        """
        return [
            {},
            {"early_tolerance": 5, "max_delay": 100},
        ]
