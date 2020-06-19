import numpy as np
import pytest
from scipy.stats import linregress
from sktime.utils._testing.forecasting import generate_polynomial_series
from sktime.utils.time_series import time_series_slope


@pytest.mark.parametrize("trend_order", [0, 3])
def test_time_series_slope_against_scipy_linregress(trend_order):
    coefs = np.random.normal(size=(trend_order + 1, 1))
    y = generate_polynomial_series(20, order=trend_order, coefs=coefs)

    # Compare with scipy's linear regression function
    x = np.arange(y.size) + 1
    a = linregress(x, y).slope
    b = time_series_slope(y)
    np.testing.assert_almost_equal(a, b, decimal=10)


# Check linear and constant cases
@pytest.mark.parametrize("slope", [-1, 0, 1])
def test_time_series_slope_against_simple_cases(slope):
    x = np.arange(1, 10)
    y = x * slope
    np.testing.assert_almost_equal(time_series_slope(y), slope, decimal=10)
