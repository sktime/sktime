"""Tests for TimeCopilotForecaster."""

__author__ = ["Hrishikesh19032004"]

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timecopilot_forecaster import TimeCopilotForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TimeCopilotForecaster),
    reason="Run test only if soft dependencies are present and incrementally",
)
def test_basic_fit_predict():
    """Test basic fit and predict."""
    y = load_airline()

    # use get_test_params LLM (TestModel)
    forecaster = TimeCopilotForecaster(
        **TimeCopilotForecaster.get_test_params()[0]
    )

    forecaster.fit(y)

    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh)

    assert y_pred is not None
    assert len(y_pred) == len(fh)


@pytest.mark.skipif(
    not run_test_for_class(TimeCopilotForecaster),
    reason="Run test only if soft dependencies are present and incrementally",
)
def test_with_component_forecasters():
    """Test passing sktime component forecasters."""
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.trend import TrendForecaster

    y = load_airline()

    base_params = TimeCopilotForecaster.get_test_params()[0]

    forecaster = TimeCopilotForecaster(
        forecasters=[NaiveForecaster(), TrendForecaster()],
        **base_params,
    )

    forecaster.fit(y)

    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh)

    assert y_pred is not None
    assert len(y_pred) == len(fh)


@pytest.mark.skipif(
    not run_test_for_class(TimeCopilotForecaster),
    reason="Run test only if soft dependencies are present and incrementally",
)
def test_query_response_after_predict():
    """Test that query response is accessible after predict."""
    y = load_airline()

    base_params = TimeCopilotForecaster.get_test_params()[0]

    forecaster = TimeCopilotForecaster(
        query="What is the expected trend?",
        **base_params,
    )

    forecaster.fit(y)
    forecaster.predict(fh=[1, 2, 3])

    response = forecaster.get_user_query_response()

    assert response is None or isinstance(response, str)