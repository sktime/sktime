"""Template for forecasting metrics (non-hierarchical)."""

from sktime.performance_metrics.base import BaseMetric


class ForecastingMetricTemplate(BaseMetric):
    """Template class for non-hierarchical forecasting metrics.

    This template provides a minimal structure to implement forecasting metrics
    for flat (non-hierarchical) time series data.

    Developers should override the `evaluate` method.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import ForecastingMetricTemplate
    >>> metric = ForecastingMetricTemplate()
    >>> try:
    ...     metric.evaluate([1, 2, 3], [1, 2, 3])
    ... except NotImplementedError:
    ...     pass
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the forecasting metric.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.

        y_pred : array-like
            Predicted values.

        Returns
        -------
        float
            Metric value.

        Raises
        ------
        NotImplementedError
            If not implemented.
        """
        raise NotImplementedError(
            "This is a template. Please implement the evaluate method."
        )
