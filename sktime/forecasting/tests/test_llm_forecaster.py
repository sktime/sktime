"""Tests for AgenticForecaster."""

__author__ = ["Nischal1425"]

import pytest

from sktime.forecasting.llm_forecaster import AgenticForecaster
from sktime.utils.estimator_checks import check_estimator


@pytest.fixture
def airline_y():
    from sktime.datasets import load_airline

    return load_airline()


class TestAgenticForecasterMock:
    """Tests using the mock backend — no API key required."""

    def test_fit_predict_basic(self, airline_y):
        """fit + predict roundtrip with mock backend."""
        fc = AgenticForecaster(llm_backend="mock")
        fc.fit(airline_y, fh=[1, 2, 3])
        pred = fc.predict()
        assert len(pred) == 3

    def test_selected_forecaster_attribute_set_after_fit(self, airline_y):
        """selected_forecaster_ is set after fit."""
        fc = AgenticForecaster(llm_backend="mock")
        fc.fit(airline_y, fh=[1])
        assert hasattr(fc, "selected_forecaster_")
        assert isinstance(fc.selected_forecaster_, str)

    def test_selected_params_attribute_set_after_fit(self, airline_y):
        """selected_params_ is set after fit."""
        fc = AgenticForecaster(llm_backend="mock")
        fc.fit(airline_y, fh=[1])
        assert hasattr(fc, "selected_params_")
        assert isinstance(fc.selected_params_, dict)

    def test_mock_selects_naive_forecaster(self, airline_y):
        """Mock backend always selects NaiveForecaster."""
        fc = AgenticForecaster(llm_backend="mock")
        fc.fit(airline_y, fh=[1])
        assert fc.selected_forecaster_ == "NaiveForecaster"

    def test_forecaster_instance_attribute(self, airline_y):
        """forecaster_ attribute holds a fitted BaseForecaster."""
        from sktime.forecasting.base import BaseForecaster

        fc = AgenticForecaster(llm_backend="mock")
        fc.fit(airline_y, fh=[1])
        assert isinstance(fc.forecaster_, BaseForecaster)

    def test_unknown_backend_raises(self, airline_y):
        """Unknown llm_backend raises ValueError."""
        fc = AgenticForecaster(llm_backend="unknown_xyz")
        with pytest.raises(ValueError, match="Unknown llm_backend"):
            fc.fit(airline_y, fh=[1])

    def test_predict_length_matches_fh(self, airline_y):
        """Prediction length matches requested horizon length."""
        fc = AgenticForecaster(llm_backend="mock")
        fc.fit(airline_y, fh=[1, 2, 3, 4, 5])
        pred = fc.predict()
        assert len(pred) == 5


class TestAgenticForecasterParsing:
    """Tests for LLM response parsing (no API call needed)."""

    def test_parse_valid_json(self):
        fc = AgenticForecaster(llm_backend="mock")
        result = fc._parse_llm_response(
            '{"forecaster": "ExponentialSmoothing", "params": {"trend": "add"}}'
        )
        assert result["forecaster"] == "ExponentialSmoothing"
        assert result["params"] == {"trend": "add"}

    def test_parse_markdown_fenced_json(self):
        fc = AgenticForecaster(llm_backend="mock")
        text = '```json\n{"forecaster": "ARIMA", "params": {}}\n```'
        result = fc._parse_llm_response(text)
        assert result["forecaster"] == "ARIMA"

    def test_parse_invalid_falls_back_to_naive(self):
        fc = AgenticForecaster(llm_backend="mock")
        result = fc._parse_llm_response("I cannot decide.")
        assert result["forecaster"] == "NaiveForecaster"

    def test_parse_json_without_forecaster_key_falls_back(self):
        fc = AgenticForecaster(llm_backend="mock")
        result = fc._parse_llm_response('{"model": "ARIMA"}')
        assert result["forecaster"] == "NaiveForecaster"


class TestAgenticForecasterRegistry:
    """Tests for registry integration."""

    def test_class_discoverable_via_all_estimators(self):
        """AgenticForecaster appears in all_estimators output."""
        from sktime.registry import all_estimators

        names = [n for n, _ in all_estimators(estimator_types="forecaster")]
        assert "AgenticForecaster" in names

    def test_get_test_params(self):
        params = AgenticForecaster.get_test_params()
        assert isinstance(params, list)
        assert len(params) > 0
        assert "llm_backend" in params[0]
        assert params[0]["llm_backend"] == "mock"


@pytest.mark.skipif(True, reason="Run manually with API key")
class TestAgenticForecasterLive:
    """Live tests requiring actual LLM API keys — skipped in CI."""

    def test_anthropic_backend(self, airline_y):
        fc = AgenticForecaster(
            task_description="monthly airline passengers with yearly seasonality",
            llm_backend="anthropic",
        )
        fc.fit(airline_y, fh=[1, 2, 3])
        pred = fc.predict()
        assert len(pred) == 3

    def test_openai_backend(self, airline_y):
        fc = AgenticForecaster(
            task_description="monthly airline passengers",
            llm_backend="openai",
        )
        fc.fit(airline_y, fh=[1])
        pred = fc.predict()
        assert len(pred) == 1
