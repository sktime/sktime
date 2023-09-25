"""Test trend forecasters."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning", "fkiraly"]
__all__ = ["get_expected_polynomial_coefs"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.trend import PolynomialTrendForecaster, TrendForecaster
from sktime.forecasting.trend._util import _get_X_numpy_int_from_pandas
from sktime.utils._testing.forecasting import make_forecasting_problem


def test_get_X_numpy():
    """Test _get_X_numpy_int_from_pandas converts to int/float as expected."""
    y = load_airline()
    X_idx = _get_X_numpy_int_from_pandas(y.index)

    # testing pd.PeriodIndex
    # this should be a 2D np.dparray, with diffs being 1 (month)
    # because month is the periodicity
    assert isinstance(X_idx, np.ndarray)
    assert X_idx.shape == (len(y), 1)
    assert (np.diff(X_idx.reshape(-1)) == 1).all()

    y_idx_datetime = y.index.to_timestamp()
    X_idx_datetime = _get_X_numpy_int_from_pandas(y_idx_datetime)

    # testing pd.DatetimeIndex
    # this should be a 2D np.ndarray, with diffs being 28, 29, 30, or 31 (days)
    # because DatetimeIndex time stamps convert to days
    assert isinstance(X_idx_datetime, np.ndarray)
    assert X_idx_datetime.shape == (len(y), 1)
    intdiffs = (np.diff(X_idx_datetime.reshape(-1))).astype(int)
    assert np.isin(intdiffs, [30, 31, 28, 29]).all()

    # testing pd.DatetimeIndex with hourly frequency
    # diffs should be 1/24, since this is converted to float, days since 1970
    df_hourly = pd.DataFrame(
        data=[10, 5, 4, 2, 10],
        index=pd.date_range(start="2000-01-01", periods=5, freq="H"),
    )
    X_idx_hourly = _get_X_numpy_int_from_pandas(df_hourly.index)
    assert isinstance(X_idx_hourly, np.ndarray)
    assert X_idx_hourly.shape == (len(df_hourly), 1)
    hourdiffs = np.diff(X_idx_hourly.reshape(-1))
    assert np.allclose(hourdiffs, np.repeat([1 / 24], len(df_hourly) - 1))

    # testing integer index
    int_ix = pd.Index([1, 3, 5, 7])
    X_idx_int = _get_X_numpy_int_from_pandas(int_ix)

    # testing pd.IntegerIndex
    # this should be an 2D integer array with entries identical to int_ix
    assert isinstance(X_idx_int, np.ndarray)
    assert X_idx_int.shape == (len(int_ix), 1)
    assert (X_idx_int.reshape(-1) == int_ix.to_numpy()).all()


def get_expected_polynomial_coefs(y, degree, with_intercept=True):
    """Compute expected coefficients from polynomial regression."""
    t_ix = _get_X_numpy_int_from_pandas(y.index).reshape(-1)
    poly_matrix = np.vander(t_ix, degree + 1)
    if not with_intercept:
        poly_matrix = poly_matrix[:, :-1]
    return np.linalg.lstsq(poly_matrix, y.to_numpy(), rcond=None)[0]


def _test_trend(degree, with_intercept):
    """Check trend, helper function."""
    y = make_forecasting_problem()
    forecaster = PolynomialTrendForecaster(degree=degree, with_intercept=with_intercept)
    forecaster.fit(y)

    # check coefficients
    # intercept is added in reverse order
    actual = forecaster.regressor_.steps[-1][1].coef_[::-1]
    expected = get_expected_polynomial_coefs(y, degree, with_intercept)
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("with_intercept", [True, False])
def test_trend(degree, with_intercept):
    """Test PolynomialTrendForecaster coefficients."""
    _test_trend(degree, with_intercept)


# zero trend does not work without intercept
def test_zero_trend():
    """Test PolynomialTrendForecaster with degree zero."""
    _test_trend(degree=0, with_intercept=True)


def test_constant_trend():
    """Test expected output from constant trend."""
    y = pd.Series(np.arange(30))
    fh = -np.arange(30)  # in-sample fh

    forecaster = PolynomialTrendForecaster(degree=1)
    y_pred = forecaster.fit(y).predict(fh)

    np.testing.assert_array_almost_equal(y, y_pred)


def test_trendforecaster_with_datetimeindex():
    """Test PolyonmialTrendForecaster with DatetimeIndex, see #4131."""
    df = load_airline()
    df.index = df.index.to_timestamp()

    f = PolynomialTrendForecaster()
    f.fit(df)

    f = TrendForecaster()
    f.fit(df)
