"""Tests for all sktime forecasting metrics."""
import pytest

pytest.importorskip("mlflow")

import pandas as pd
import pytest

from sktime.tests.test_all_estimators import BaseFixtureGenerator, QuickTester


class ForecastingMetricFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for forecasting metric tests."""

    estimator_type_filter = "metric_forecasting"

    fixture_sequence = [
        "estimator_class",
        "estimator_instance",
        "fitted_estimator",
        "scenario",
    ]


class TestAllForecastingMetrics(
    ForecastingMetricFixtureGenerator, QuickTester
):
    """Module level tests for all sktime forecasting metrics."""

    def test_evaluate_output(self, estimator_instance):
        """Test expected output type of evaluate_output."""

        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        y_pred = pd.Series([1.1, 1.9, 3.2, 3.8])

        metric = estimator_instance
        requires_y_true = metric.get_tag("requires_y_true")

        # standard case
        loss = metric(y_true, y_pred)
        assert isinstance(loss, float)

        # test without y_true
        if not requires_y_true:
            loss1 = metric(y_pred=y_pred)
            loss2 = metric(y_pred)

            assert isinstance(loss1, float)
            assert isinstance(loss2, float)
            assert loss1 == loss2
        else:
            with pytest.raises(TypeError):
                metric(y_pred)

            with pytest.raises(TypeError):
                metric(y_pred=y_pred)