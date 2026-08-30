from sktime.performance_metrics.detection._base import BaseDetectionMetric


class DetectionDelayMean(BaseDetectionMetric):
    """Mean delay of detected events (penalizes only late detections)."""

    _tags = {
        "object_type": ["metric_detection", "metric"],
        "scitype:y": "points",
        "requires_X": False,
        "requires_y_true": True,
        "lower_is_better": True,  # lower delay is better
    }

    def _evaluate(self, y_true, y_pred, X=None):
        if len(y_true) == 0 or len(y_pred) == 0:
            return float("inf")

        true_ilocs = sorted(y_true["ilocs"].values)
        pred_ilocs = sorted(y_pred["ilocs"].values)

        n = min(len(true_ilocs), len(pred_ilocs))

        delays = []
        for i in range(n):
            delay = pred_ilocs[i] - true_ilocs[i]
            delays.append(max(0, delay))  # only penalize late

        return sum(delays) / n
