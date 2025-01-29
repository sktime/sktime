import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric


class TimeSeriesAUPRC(BaseDetectionMetric):
    """Compute the area under the precision-recall curve for time series."""

    def __init__(self, integration="trapezoid", weighted_precision=True):
        self.integration = integration
        self.weighted_precision = weighted_precision

        super().__init__()

    def ts_auprc(self, y_true, scores, X):
        pass

    def _evaluate(self, y_true, scores, X):
        thresholds = np.unique(scores)
        precision = np.empty(len(thresholds) + 1)
        recall = np.empty(len(thresholds) + 1)
        # predictions = np.empty_like(scores, dtype=int)

        precision[-1] = 1
        recall[-1] = 0
