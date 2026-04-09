from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric
import numpy as np


class HierarchicalForecastingMetric(BaseForecastingErrorMetric):
    """Forecasting metric for hierarchical time series using MAE.

    This metric supports hierarchical forecasting scenarios,
    where data may have multiple aggregation levels
    (e.g., city → state → country).
    """

    _tags = {
        "authors": ["your-github-id"],
        "object_type": ["metric_forecasting", "metric"],
        "requires_y_true": True,
        "lower_is_better": True,
    }

    def __init__(self):
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute MAE across hierarchical levels."""

        import numpy as np

        # Convert to numpy (works for simple + hierarchical flattened data)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        return np.mean(np.abs(y_true - y_pred))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{}]