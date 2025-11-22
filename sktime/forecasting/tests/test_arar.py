#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for ARAR forecaster."""

__author__ = ["Akai01"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.arar import ARARForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_simple():
    """Test ARAR forecaster with simple time series."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum())

    # Fit the forecaster
    forecaster = ARARForecaster()
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3, 4, 5]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_with_params():
    """Test ARAR forecaster with custom parameters."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum())

    # Fit the forecaster with custom parameters
    forecaster = ARARForecaster(max_ar_depth=10, max_lag=15)
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_prediction_intervals():
    """Test ARAR forecaster prediction intervals."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum())

    # Fit the forecaster
    forecaster = ARARForecaster()
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
        assert all(pred_int[(0, cov, "lower")] <= pred_int[(0, cov, "upper")])


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_quantiles():
    """Test ARAR forecaster quantile predictions."""
    # Create a simple time series
    y = pd.Series(np.random.randn(50).cumsum())

    # Fit the forecaster
    forecaster = ARARForecaster()
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
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_short_series():
    """Test ARAR forecaster with short time series."""
    # Create a very short time series (should trigger warning)
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Fit the forecaster (should use safe mode and return mean fallback)
    forecaster = ARARForecaster(safe=True)
    forecaster.fit(y)

    # Make predictions
    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))


@pytest.mark.skipif(
    not run_test_for_class(ARARForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_arar_forecaster_airline_data():
    """Test ARAR forecaster with airline data."""
    from sktime.datasets import load_airline

    y = load_airline()

    # Fit the forecaster
    forecaster = ARARForecaster()
    forecaster.fit(y)

    # Make predictions
    fh = list(range(1, 13))  # 12 months ahead
    y_pred = forecaster.predict(fh=fh)

    # Check predictions
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert all(np.isfinite(y_pred))

    # Check that predictions are reasonable (positive for airline data)
    assert all(y_pred > 0)
