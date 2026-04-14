"""Mean detection delay metric for time series event detection."""

import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class DetectionDelayMean(BaseDetectionMetric):
    r"""Mean detection delay for event detection tasks.

    For each true event, finds the earliest predicted detection at or after it,
    then computes the delay. Early detections can be clipped via ``early_tolerance``.

    This metric is lower-is-better. It is suitable when there can be multiple
    events and multiple detections in a time series.

    Parameters
    ----------
    early_tolerance : int, default=0
        Tolerated earliness in steps. Delays >= -early_tolerance are clipped to 0.
    max_delay : int or None, default=None
        If not None, delays larger than this value are capped at ``max_delay``.
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": True,  # lower mean delay is better
    }

    def __init__(self, early_tolerance=0, max_delay=None):
        super().__init__()
        self.early_tolerance = early_tolerance
        self.max_delay = max_delay

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute mean detection delay across true events."""
        if len(y_true) == 0:
            return 0.0

        true_ilocs = np.sort(y_true["ilocs"].values)

        if len(y_pred) == 0:
            # Penalty when no predictions at all
            penalty = self.max_delay if self.max_delay is not None else 1000.0
            return float(penalty)

        pred_ilocs = np.sort(y_pred["ilocs"].values)
        delays = []

        for t in true_ilocs:
            # Find first prediction at or after this true event
            later = pred_ilocs[pred_ilocs >= t]
            if len(later) == 0:
                delay = self.max_delay if self.max_delay is not None else 1000.0
            else:
                delay = later[0] - t

            # Clip early detections
            delay = max(delay, -self.early_tolerance)
            delay = max(0, delay)  # early detections contribute 0 delay by default

            if self.max_delay is not None:
                delay = min(delay, self.max_delay)

            delays.append(delay)

        return float(np.mean(delays))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        return [
            {},
            {"early_tolerance": 5, "max_delay": 100},
        ]
