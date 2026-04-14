"""Mean detection delay for time series event detection tasks."""

import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class DetectionDelayMean(BaseDetectionMetric):
    r"""Mean detection delay metric for event detection.

    For each true event, this metric finds the earliest unused prediction
    that occurs within ``early_tolerance`` steps **before** or any time
    **after** the true event. The delay is computed as ``pred_time - true_time``.
    Detections within the tolerance window are clipped to delay = 0.

    Unmatched true events (no suitable prediction found) receive a penalty
    equal to ``max_delay`` (or 1000.0 by default).

    It uses **greedy earliest-available matching**, which works well for
    typical cases with multiple events and multiple predictions in one series.

    Lower values are better (faster average detection).

    Parameters
    ----------
    early_tolerance : int, default=0
        Maximum steps a prediction can precede a true event and still be
        considered timely (delay is set to 0). Useful when slightly early
        alerts are acceptable or beneficial.
    max_delay : int or None, default=None
        Cap for the delay of any detection. Also serves as the penalty value
        for unmatched (missed) true events. If None, unmatched events get
        penalty = 1000.0.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.performance_metrics.detection import DetectionDelayMean
    >>>
    >>> y_true = pd.DataFrame({"ilocs": [100]})
    >>> y_pred = pd.DataFrame({"ilocs": [110]})
    >>> metric = DetectionDelayMean()
    >>> metric(y_true, y_pred)  # delayed by 10
    10.0
    >>>
    >>> # Early detection within tolerance
    >>> metric = DetectionDelayMean(early_tolerance=5)
    >>> y_pred_early = pd.DataFrame({"ilocs": [96]})
    >>> metric(y_true, y_pred_early)
    0.0
    """

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": True,
    }

    def __init__(self, early_tolerance: int = 0, max_delay: int | None = None):
        super().__init__()
        self.early_tolerance = early_tolerance
        self.max_delay = max_delay

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute mean detection delay using greedy matching."""
        if len(y_true) == 0:
            return 0.0

        true_ilocs = np.sort(y_true["ilocs"].values)
        pred_ilocs = (
            np.sort(y_pred["ilocs"].values)
            if len(y_pred) > 0
            else np.array([], dtype=int)
        )

        delays = []
        used = np.zeros(len(pred_ilocs), dtype=bool)

        for t in true_ilocs:
            # Find earliest unused prediction in the allowed window
            candidates = np.where((pred_ilocs >= t - self.early_tolerance) & (~used))[0]

            if len(candidates) == 0:
                # Missed event → penalty
                delay = self.max_delay if self.max_delay is not None else 1000.0
            else:
                idx = candidates[0]
                delay = pred_ilocs[idx] - t
                used[idx] = True

            # Early detections (within tolerance) contribute 0 delay
            delay = max(0, delay)

            # Cap at max_delay if provided
            if self.max_delay is not None:
                delay = min(delay, self.max_delay)

            delays.append(delay)

        return float(np.mean(delays))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {},
            {"early_tolerance": 5, "max_delay": 100},
        ]
