#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for ETSForecaster."""

__author__ = ["resul.akay@taf-society.org"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.ets import ETSForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_simple():
    """Test ETSForecaster with simple exponential smoothing (ANN)."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum() + 100)

    # Fit simple exponential smoothing (ANN)
    forecaster = ETSForecaster(model="ANN")
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3, 4, 5]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_trend():
    """Test ETSForecaster with Holt's linear trend (AAN)."""
    # Create a time series with trend
    y = pd.Series(np.arange(50) * 2 + np.random.randn(50) * 5 + 100)

    # Fit Holt's linear method (AAN)
    forecaster = ETSForecaster(model="AAN")
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_damped():
    """Test ETSForecaster with damped trend (AAdN)."""
    # Create a time series with trend
    y = pd.Series(np.arange(50) * 2 + np.random.randn(50) * 5 + 100)

    # Fit damped trend model (AAN with damped=True)
    forecaster = ETSForecaster(model="AAN", damped=True)
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_seasonal():
    """Test ETSForecaster with additive Holt-Winters (AAA)."""
    # Use airline data which has seasonality
    y = load_airline()

    # Fit additive Holt-Winters (AAA)
    forecaster = ETSForecaster(m=12, model="AAA")
    forecaster.fit(y)

    # Make predictions
    fh = list(range(1, 13))  # 12 months ahead
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))
    # Airline data is always positive
    assert all(y_pred > 0)


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_multiplicative():
    """Test ETSForecaster with multiplicative Holt-Winters (MAM)."""
    # Use airline data
    y = load_airline()

    # Fit multiplicative Holt-Winters (MAM)
    forecaster = ETSForecaster(m=12, model="MAM")
    forecaster.fit(y)

    # Make predictions
    fh = list(range(1, 7))  # 6 months ahead
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))
    assert all(y_pred > 0)


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_auto():
    """Test ETSForecaster with automatic model selection (ZZZ)."""
    # Use airline data
    y = load_airline()

    # Fit with automatic model selection
    forecaster = ETSForecaster(m=12, model="ZZZ")
    forecaster.fit(y)

    # Make predictions
    fh = list(range(1, 7))
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))

    # Check that a model was selected
    assert hasattr(forecaster, "model_")
    assert forecaster.model_ is not None


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_prediction_intervals():
    """Test ETSForecaster prediction intervals."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum() + 100)

    # Fit the forecaster
    forecaster = ETSForecaster(model="AAN")
    forecaster.fit(y)

    # Make predictions with intervals
    fh = [1, 2, 3]
    coverage = [0.80, 0.90]
    pred_int = forecaster.predict_interval(fh=fh, coverage=coverage)

    # Check prediction intervals
    assert isinstance(pred_int, pd.DataFrame)
    assert pred_int.shape[0] == len(fh)
    assert pred_int.shape[1] == len(coverage) * 2  # lower and upper for each coverage

    # Check that lower bounds are less than upper bounds
    for cov in coverage:
        assert all(pred_int[(cov, "lower")] <= pred_int[(cov, "upper")])


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_quantiles():
    """Test ETSForecaster quantile predictions."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum() + 100)

    # Fit the forecaster
    forecaster = ETSForecaster(model="ANN")
    forecaster.fit(y)

    # Make quantile predictions
    fh = [1, 2, 3]
    alpha = [0.05, 0.5, 0.95]
    quantiles = forecaster.predict_quantiles(fh=fh, alpha=alpha)

    # Check quantiles
    assert isinstance(quantiles, pd.DataFrame)
    assert quantiles.shape[0] == len(fh)

    # Check that quantiles are ordered (0.05 <= 0.5 <= 0.95)
    var_name = y.name if y.name is not None else 0
    assert all(quantiles[(var_name, 0.05)] <= quantiles[(var_name, 0.5)])
    assert all(quantiles[(var_name, 0.5)] <= quantiles[(var_name, 0.95)])


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_with_fixed_params():
    """Test ETSForecaster with fixed smoothing parameters."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum() + 100)

    # Fit with fixed alpha parameter
    forecaster = ETSForecaster(model="ANN", alpha=0.3)
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))

    # Check that the fitted alpha is close to the fixed value
    assert hasattr(forecaster, "model_")
    assert abs(forecaster.model_.params.alpha - 0.3) < 0.01


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_boxcox():
    """Test ETSForecaster with Box-Cox transformation."""
    # Use airline data which has increasing variance
    y = load_airline()

    # Fit with Box-Cox transformation
    forecaster = ETSForecaster(m=12, model="AAN", lambda_auto=True)
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))
    assert all(y_pred > 0)

    # Check that transformation was applied
    assert hasattr(forecaster, "model_")
    assert forecaster.model_.transform is not None


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_short_series():
    """Test ETSForecaster with short time series."""
    # Create a short time series
    y = pd.Series([10.0, 12.0, 11.0, 13.0, 14.0, 15.0, 16.0, 18.0])

    # Fit simple model for short series
    forecaster = ETSForecaster(model="ANN")
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_bounds():
    """Test ETSForecaster with different bounds specifications."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum() + 100)

    # Test with different bounds
    for bounds in ["usual", "admissible", "both"]:
        forecaster = ETSForecaster(model="AAN", bounds=bounds)
        forecaster.fit(y)

        fh = [1, 2, 3]
        y_pred = forecaster.predict(fh=fh)

        assert isinstance(y_pred, pd.Series)
        assert len(y_pred) == len(fh)
        assert all(np.isfinite(y_pred))


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_get_test_params():
    """Test ETSForecaster get_test_params method."""
    params = ETSForecaster.get_test_params()

    # Should return a list of parameter dictionaries
    assert isinstance(params, list)
    assert len(params) > 0

    # Each should be a dict
    for param_dict in params:
        assert isinstance(param_dict, dict)

        # Should be able to instantiate with these parameters
        forecaster = ETSForecaster(**param_dict)
        assert forecaster is not None


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_constant_series():
    """Test ETSForecaster with constant series."""
    # Create a constant time series
    y = pd.Series([10.0] * 20)

    # Fit the forecaster - should handle constant series gracefully
    forecaster = ETSForecaster(model="ANN")
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))
    # For constant series, predictions should be close to the constant value
    assert all(np.abs(y_pred - 10.0) < 0.1)


@pytest.mark.skipif(
    not run_test_for_class(ETSForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ets_forecaster_different_horizons():
    """Test ETSForecaster with different forecast horizons."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum() + 100)

    # Fit the forecaster
    forecaster = ETSForecaster(model="AAN")
    forecaster.fit(y)

    # Test with different horizon specifications
    fh_list = [
        [1],  # Single step
        [1, 2, 3],  # Multiple steps
        list(range(1, 11)),  # 10 steps ahead
        [3, 5, 7],  # Non-consecutive steps
    ]

    for fh in fh_list:
        y_pred = forecaster.predict(fh=fh)
        assert isinstance(y_pred, pd.Series)
        assert len(y_pred) == len(fh)
        assert all(np.isfinite(y_pred))
