"""Tests for time series utilities."""
import numpy as np
import pytest
from scipy.stats import linregress

from sktime.utils._testing.forecasting import _generate_polynomial_series
from sktime.utils.slope_and_trend import _fit_trend, _slope


@pytest.mark.parametrize("trend_order", [0, 3])
def test_time_series_slope_against_scipy_linregress(trend_order):
    """Test time series slope against scipy lingress."""
    coefs = np.random.normal(size=(trend_order + 1, 1))
    y = _generate_polynomial_series(20, order=trend_order, coefs=coefs)

    # Compare with scipy's linear regression function
    x = np.arange(y.size) + 1
    a = linregress(x, y).slope
    b = _slope(y)
    np.testing.assert_almost_equal(a, b, decimal=10)


# Check linear and constant cases
@pytest.mark.parametrize("slope", [-1, 0, 1])
def test_time_series_slope_against_simple_cases(slope):
    """Test time series slope against simple cases."""
    x = np.arange(1, 10)
    y = x * slope
    np.testing.assert_almost_equal(_slope(y), slope, decimal=10)


@pytest.mark.parametrize("order", [0, 1, 2])  # polynomial order
@pytest.mark.parametrize("n_timepoints", [1, 10])  # number of time series observations
@pytest.mark.parametrize("n_instances", [1, 10])  # number of samples
def test_fit_remove_add_trend(order, n_instances, n_timepoints):
    """Test fitted coefficients."""
    coefs = np.random.normal(size=order + 1).reshape(-1, 1)
    x = np.column_stack(
        [
            _generate_polynomial_series(n_timepoints, order, coefs=coefs)
            for _ in range(n_instances)
        ]
    ).T
    # assert x.shape == (n_samples, n_obs)

    # check shape of fitted coefficients
    coefs = _fit_trend(x, order=order)
    assert coefs.shape == (n_instances, order + 1)
