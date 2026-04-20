"""Template for hierarchical forecasting metrics."""

from sktime.performance_metrics.base import BaseMetric


class ForecastingHierarchicalMetricTemplate(BaseMetric):
    """Template class for hierarchical forecasting metrics.

    This template is intended for metrics that operate on hierarchical
    or grouped time series.

    Developers should override the `evaluate` method.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import (
    ...     ForecastingHierarchicalMetricTemplate
    ... )
    >>> metric = ForecastingHierarchicalMetricTemplate()
    >>> try:
    ...     metric.evaluate([1, 2], [1, 2])
    ... except NotImplementedError:
    ...     pass
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, y_true, y_pred, hierarchy=None, **kwargs):
        """Evaluate hierarchical forecasting metric.

        Parameters
        ----------
        y_true : array-like
        y_pred : array-like
        hierarchy : optional

        Returns
        -------
        float
            Metric value.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "This is a template. Please implement the evaluate method."
        )
