# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for AdvancedTabICLForecaster."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.advanced_tabicl import AdvancedTabICLForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(AdvancedTabICLForecaster),
    reason="run test only if soft dependencies are present and incrementally",
)


class _DummyTabICLRegressor:
    """Simple deterministic stand-in for TabICLRegressor in unit tests."""

    def fit(self, X, y):
        self._is_fitted = True
        self._last_seen_target = float(np.asarray(y).reshape(-1)[-1])
        return self

    def predict(self, X):
        x_arr = np.asarray(X, dtype=float)
        return np.array([x_arr[0, -1] + 1.0], dtype=float)


@pytest.fixture
def patch_dummy_tabicl(monkeypatch):
    """Patch model initialization to avoid external dependency behavior in tests."""

    def _dummy_init_model(self):
        return _DummyTabICLRegressor()

    monkeypatch.setattr(AdvancedTabICLForecaster, "_init_model", _dummy_init_model)


@pytest.fixture
def y_series():
    """Create a simple deterministic univariate series."""
    values = np.arange(20, dtype=float)
    return pd.Series(values, index=pd.RangeIndex(20), name="y")


def test_basic_fit_predict_returns_series(y_series, patch_dummy_tabicl):
    """Basic fit/predict should run and return pandas Series."""
    forecaster = AdvancedTabICLForecaster(window_length=5, strategy="recursive")
    fh = ForecastingHorizon([1], is_relative=True)

    forecaster.fit(y_series)
    y_pred = forecaster.predict(fh=fh)

    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)


def test_multistep_forecast_length_matches_fh(y_series, patch_dummy_tabicl):
    """Multi-step forecasts should have same length as forecasting horizon."""
    forecaster = AdvancedTabICLForecaster(window_length=5, strategy="recursive")
    fh = ForecastingHorizon([1, 2, 4, 6], is_relative=True)

    forecaster.fit(y_series)
    y_pred = forecaster.predict(fh=fh)

    assert len(y_pred) == len(fh)


def test_recursive_strategy_updates_window(y_series, patch_dummy_tabicl):
    """Recursive strategy should feed predictions back into the next step input."""
    forecaster = AdvancedTabICLForecaster(window_length=5, strategy="recursive")
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    forecaster.fit(y_series)
    y_pred = forecaster.predict(fh=fh)

    expected = np.array([20.0, 21.0, 22.0])
    assert np.allclose(y_pred.to_numpy(), expected)


def test_direct_strategy_predicts_multistep(y_series, patch_dummy_tabicl):
    """Direct strategy should fit and predict for provided horizon steps."""
    forecaster = AdvancedTabICLForecaster(window_length=5, strategy="direct")
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    forecaster.fit(y_series, fh=fh)
    y_pred = forecaster.predict(fh=fh)

    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)


def test_window_length_larger_than_data_raises(patch_dummy_tabicl):
    """Fitting with too-large window_length should raise ValueError."""
    y = pd.Series(np.arange(5, dtype=float), index=pd.RangeIndex(5), name="y")
    forecaster = AdvancedTabICLForecaster(window_length=10, strategy="recursive")

    with pytest.raises(ValueError, match="window_length"):
        forecaster.fit(y)


def test_prediction_index_aligns_with_fh(y_series, patch_dummy_tabicl):
    """Prediction index should match absolute forecasting horizon index exactly."""
    forecaster = AdvancedTabICLForecaster(window_length=5, strategy="recursive")
    fh = ForecastingHorizon([1, 3, 5], is_relative=True)

    forecaster.fit(y_series)
    y_pred = forecaster.predict(fh=fh)

    expected_index = fh.to_absolute(forecaster.cutoff).to_pandas()
    assert y_pred.index.equals(expected_index)
