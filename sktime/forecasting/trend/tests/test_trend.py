"""Test trend forecasters."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning", "fkiraly"]
__all__ = ["get_expected_polynomial_coefs"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster, TrendForecaster
from sktime.forecasting.trend._util import _get_X_numpy_int_from_pandas
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.forecasting import make_forecasting_problem


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("with_intercept", [True, False])
def test_trend(degree, with_intercept):
    """Test PolynomialTrendForecaster coefficients."""
    _test_trend(degree, with_intercept)


# zero trend does not work without intercept
@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_zero_trend():
    """Test PolynomialTrendForecaster with degree zero."""
    _test_trend(degree=0, with_intercept=True)


@pytest.mark.skipif(
    not run_test_for_class(PolynomialTrendForecaster, _get_X_numpy_int_from_pandas),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_constant_trend():
    """Test expected output from constant trend."""
    y = pd.Series(np.arange(30))
    fh = -np.arange(30)  # in-sample fh

    forecaster = PolynomialTrendForecaster(degree=1)
    y_pred = forecaster.fit(y).predict(fh)

    np.testing.assert_array_almost_equal(y, y_pred)


@pytest.mark.skipif(
    not run_test_for_class(
        [PolynomialTrendForecaster, TrendForecaster, _get_X_numpy_int_from_pandas]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_trendforecaster_with_datetimeindex():
    """Test PolyonmialTrendForecaster with DatetimeIndex, see #4131."""
    df = load_airline()
    df.index = df.index.to_timestamp()

    f = PolynomialTrendForecaster()
    f.fit(df)

    f = TrendForecaster()
    f.fit(df)


def test_predict_var():
    """Unit test for predict_interval method in PolynomialTrendForecaster.
    We only need to test _predict_var() since the residuals are assumed normal.
    The test:
    Create series IID N(0,1), do a quadratic fit, and confirm Var(residuals) ~ 1
    """
    N = 500
    data = np.random.normal(0, 1, N)
    date_index = pd.date_range(start="1970-01-01", periods=N, freq="D")
    df = pd.DataFrame(data, index=date_index, columns=["x"])
    df.at[date_index[42], "x"] = 1
    df.at[date_index[43], "x"] = -1

    forecaster = PolynomialTrendForecaster(degree=2, prediction_intervals=False)
    forecaster.fit(df)
    residuals = forecaster.predict_residuals(y=df)
    s = np.std(residuals)
    y = df / s

    forecaster = PolynomialTrendForecaster(degree=2, prediction_intervals=True)
    forecaster.fit(y)
    fh = ForecastingHorizon([1, 2, 3, 4, 5, 6], is_relative=True)
    a = forecaster.predict_var(fh=fh)
    np.testing.assert_allclose(a.iloc[0, 0], 1.0, rtol=0.05)
