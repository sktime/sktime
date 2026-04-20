import pytest


def test_forecasting_metric_template():
    from sktime.performance_metrics.forecasting import ForecastingMetricTemplate

    metric = ForecastingMetricTemplate()

    with pytest.raises(NotImplementedError):
        metric.evaluate([1, 2, 3], [1, 2, 3])


def test_forecasting_hierarchical_metric_template():
    from sktime.performance_metrics.forecasting import (
        ForecastingHierarchicalMetricTemplate,
    )

    metric = ForecastingHierarchicalMetricTemplate()

    with pytest.raises(NotImplementedError):
        metric.evaluate([1, 2, 3], [1, 2, 3])
