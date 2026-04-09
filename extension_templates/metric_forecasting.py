from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric
import numpy as np


class MyForecastingMetric(BaseForecastingErrorMetric):
    """Simple forecasting metric using Mean Absolute Error (MAE)."""

    _tags = {
        "authors": ["your-github-id"],
        "object_type": ["metric_forecasting", "metric"],
        "requires_y_true": True,
        "lower_is_better": True,
    }

    def __init__(self):
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """Compute MAE between true and predicted values."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{}]