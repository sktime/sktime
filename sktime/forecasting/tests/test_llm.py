"""Tests for LLMForecaster."""

import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.llm import (
    DummyLLM,
    DummyLLMGarbage,
    LLMForecaster,
)
from sktime.forecasting.naive import NaiveForecaster


def test_llm_forecaster_fit_predict():
    y = pd.Series([1, 2, 3, 4, 5], index=pd.RangeIndex(5))
    fh = ForecastingHorizon([1, 2], is_relative=True)

    forecaster = LLMForecaster(
        llm=DummyLLM("FORECASTER: naive\nREASON: simple baseline"),
        candidate_forecasters=(("naive", NaiveForecaster()),),
    )

    forecaster.fit(y, fh=fh)
    y_pred = forecaster.predict(fh=fh)

    assert isinstance(y_pred, pd.Series)
    assert forecaster.selected_forecaster_ == "naive"
    assert hasattr(forecaster, "last_response_")


def test_llm_forecaster_fallback_on_invalid_response():
    y = pd.Series([10, 12, 13, 15, 18], index=pd.RangeIndex(5))
    fh = ForecastingHorizon([1, 2], is_relative=True)

    forecaster = LLMForecaster(
        llm=DummyLLMGarbage(),
        candidate_forecasters=(("naive", NaiveForecaster()),),
        default_forecaster=NaiveForecaster(),
    )

    forecaster.fit(y, fh=fh)
    y_pred = forecaster.predict(fh=fh)

    assert isinstance(y_pred, pd.Series)
    assert forecaster.selected_forecaster_ == "NaiveForecaster"


def test_llm_forecaster_invalid_strategy():
    y = pd.Series([1, 2, 3, 4], index=pd.RangeIndex(4))
    fh = ForecastingHorizon([1], is_relative=True)

    forecaster = LLMForecaster(
        llm=DummyLLM(),
        strategy="unsupported",
    )

    with pytest.raises(ValueError, match="Only 'select' is supported"):
        forecaster.fit(y, fh=fh)
