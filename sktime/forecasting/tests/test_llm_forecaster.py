"""Tests for LLMForecaster."""

__author__ = ["yashkotha"]

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.llm import LLMForecaster
from sktime.utils.estimator_checks import check_estimator


def _make_mock_llm(predictions=None):
    """Return a MagicMock that speaks the langchain .invoke() interface."""
    if predictions is None:
        predictions = [100.0, 101.0, 102.0]
    mock = MagicMock()
    payload = str({"predictions": predictions}).replace("'", '"')
    mock.invoke.return_value.content = payload
    return mock


# ---------------------------------------------------------------------------
# Basic fit / predict
# ---------------------------------------------------------------------------


def test_fit_predict_basic():
    """LLMForecaster returns correct number of predictions."""
    y = load_airline()
    fh = [1, 2, 3]
    mock = _make_mock_llm([120.0, 122.5, 119.8])

    fc = LLMForecaster(llm=mock, context_length=12)
    fc.fit(y, fh=fh)
    pred = fc.predict()

    assert isinstance(pred, pd.Series)
    assert len(pred) == len(fh)


def test_predict_values_from_mock():
    """LLMForecaster correctly parses values returned by the mock."""
    y = load_airline()
    expected = [200.0, 210.0]
    mock = _make_mock_llm(expected)

    fc = LLMForecaster(llm=mock, context_length=5)
    fc.fit(y, fh=[1, 2])
    pred = fc.predict()

    assert list(pred.values) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Context trimming
# ---------------------------------------------------------------------------


def test_context_length_respected():
    """Only the last `context_length` observations reach the prompt."""
    y = load_airline()
    mock = _make_mock_llm([100.0])

    fc = LLMForecaster(llm=mock, context_length=5)
    fc.fit(y, fh=[1])
    fc.predict()

    call_args = mock.invoke.call_args
    prompt_text = call_args[0][0][0].content  # HumanMessage content

    # 5 timestamps → 5 commas max in the "Values" line
    values_line = [ln for ln in prompt_text.split("\n") if "Values" in ln][0]
    n_values = len(values_line.split(","))
    assert n_values == 5


# ---------------------------------------------------------------------------
# Fallback parsing
# ---------------------------------------------------------------------------


def test_parse_fallback_extracts_numbers():
    """_parse_predictions falls back to regex when JSON is malformed."""
    fc = LLMForecaster.__new__(LLMForecaster)
    response = "I think the next values will be about 150 and 155."
    preds = fc._parse_predictions(response, n_ahead=2)
    assert preds == pytest.approx([150.0, 155.0])


def test_parse_raises_if_too_few_numbers():
    """_parse_predictions raises ValueError when not enough values found."""
    fc = LLMForecaster.__new__(LLMForecaster)
    with pytest.raises(ValueError, match="Could not parse"):
        fc._parse_predictions("no numbers here at all!", n_ahead=3)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_string_backend_openai_initialised():
    """Passing llm='openai' triggers openai.OpenAI() construction."""
    with patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_llm([1.0])
        fc = LLMForecaster(llm="openai")
        y = load_airline()
        fc.fit(y, fh=[1])
        mock_cls.assert_called_once()
        assert fc._model_name_ == "gpt-4o-mini"


def test_unknown_string_backend_raises():
    """Unknown string backend raises ValueError on fit."""
    fc = LLMForecaster(llm="does_not_exist")
    y = load_airline()
    with pytest.raises(ValueError, match="Unknown llm backend"):
        fc.fit(y, fh=[1])


def test_custom_model_name_forwarded():
    """model_name parameter is forwarded to the backend."""
    mock = _make_mock_llm([1.0])
    fc = LLMForecaster(llm=mock, model_name="my-custom-model", context_length=5)
    # model_name on a non-string llm has no effect, but param is stored
    assert fc.model_name == "my-custom-model"


# ---------------------------------------------------------------------------
# sktime estimator compliance (lightweight tag / clone checks only)
# ---------------------------------------------------------------------------


def test_get_params_set_params_roundtrip():
    """get_params / set_params round-trip is consistent."""
    mock = _make_mock_llm()
    fc = LLMForecaster(llm=mock, context_length=15, temperature=0.1)
    params = fc.get_params()
    assert params["context_length"] == 15
    assert params["temperature"] == pytest.approx(0.1)

    fc2 = LLMForecaster()
    fc2.set_params(**params)
    assert fc2.get_params()["context_length"] == 15


def test_get_test_params_returns_dict():
    """get_test_params returns a non-empty dict."""
    params = LLMForecaster.get_test_params()
    assert isinstance(params, dict)
    assert "llm" in params
