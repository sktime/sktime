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
        index=pd.date_range(start="2000-01-01", periods=5, freq="h"),
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


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_get_X_numpy_datetime_resolution(unit):
    """Test _get_X_numpy_int_from_pandas does not depend on datetime resolution."""
    timestamps = np.array(
        ["2000-01-01", "2000-01-02", "2000-01-04"], dtype=f"datetime64[{unit}]"
    )
    X = _get_X_numpy_int_from_pandas(pd.DatetimeIndex(timestamps))
    np.testing.assert_array_equal(X.reshape(-1), [10957.0, 10958.0, 10960.0])


def get_expected_polynomial_coefs(y, degree, with_intercept=True):
    """Compute expected coefficients from polynomial regression.

    Coefficients are in units of the time index, in decreasing order of powers,
    i.e., the same order as ``np.polyfit``.

    The least squares problem is solved on a scaled time axis, to avoid the
    ill-conditioning of the Vandermonde matrix for large time stamps,
    and the coefficients are converted back to the original units afterwards.
    """
    t_ix = _get_X_numpy_int_from_pandas(y.index).reshape(-1).astype("float64")
    if with_intercept:
        # fit in the domain [-1, 1], then convert back to original time units
        poly = np.polynomial.Polynomial.fit(t_ix, y.to_numpy(), degree)
        return poly.convert().coef[::-1]

    # without intercept, the polynomial must vanish at t=0, so we only scale.
    # The least squares solution is computed by sklearn, so that the
    # treatment of nearly collinear features (which can occur for short series
    # far away from t=0) is consistent with the forecaster
    from sklearn.linear_model import LinearRegression

    scale = np.max(np.abs(t_ix))
    powers = np.arange(1, degree + 1)
    poly_matrix = (t_ix.reshape(-1, 1) / scale) ** powers
    coefs = LinearRegression(fit_intercept=False).fit(poly_matrix, y.to_numpy()).coef_
    return (coefs / scale**powers)[::-1]


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


@pytest.mark.xfail(reason="known bug, needs to be investigated, see #11495")
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


@pytest.mark.skipif(
    not run_test_for_class(
        [PolynomialTrendForecaster, TrendForecaster, _get_X_numpy_int_from_pandas]
    ),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("freq", ["D", "h"])
def test_polynomial_trend_large_time_stamps(freq):
    """Test numerical stability for large time stamps and higher degree, see #8686.

    Time stamps in days since 1970 are large numbers, so that the polynomial
    features of degree 2 or higher are severely ill-conditioned when the
    regression is carried out on the raw time stamps. In that case, degrees 2, 3
    and 4 used to yield the same, wrong, fit.
    """
    index = pd.date_range("2019-01-01", "2024-12-31", freq=freq)
    n = len(index)
    t = np.linspace(-1, 1, n)
    trend = 1000 + 500 * t + 200 * t**2 - 300 * t**3
    rng = np.random.default_rng(42)
    y = pd.Series(trend + rng.normal(0, 20, n), index=index)
    fh = ForecastingHorizon(index, is_relative=False)
    X_orig = _get_X_numpy_int_from_pandas(index)

    y_pred = {}
    for degree in [2, 3, 4]:
        forecaster = PolynomialTrendForecaster(degree=degree)
        forecaster.fit(y)
        y_pred[degree] = forecaster.predict(fh).to_numpy()

        # coefficients are in units of the original time stamps, see docstring,
        # and must agree with a numerically stable reference fit
        actual = forecaster.regressor_.steps[-1][1].coef_[::-1]
        expected = get_expected_polynomial_coefs(y, degree, with_intercept=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-12)

        # the fitted pipeline, applied to original time stamps, agrees with predict
        np.testing.assert_allclose(
            forecaster.regressor_.predict(X_orig), y_pred[degree], rtol=1e-8
        )

    # the fitted cubic trend recovers the true cubic trend up to noise
    rmse = np.sqrt(np.mean((y_pred[3] - trend) ** 2))
    assert rmse < 5

    # different degrees yield different fits
    assert not np.allclose(y_pred[2], y_pred[3])
    assert not np.allclose(y_pred[3], y_pred[4], rtol=1e-6)


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_polynomial_trend_large_time_stamps_no_intercept():
    """Test fitting with large time stamps and ``with_intercept=False``, see #8686.

    Without intercept, the polynomial must vanish at the origin of the time axis,
    so time is scaled but not shifted internally. The coefficients must still be
    in original time units, and the fitted pipeline must agree with ``predict``.
    """
    index = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    n = len(index)
    t = np.linspace(-1, 1, n)
    rng = np.random.default_rng(42)
    y = pd.Series(1000 + 500 * t + 200 * t**2 + rng.normal(0, 20, n), index=index)
    fh = ForecastingHorizon(index, is_relative=False)
    X_orig = _get_X_numpy_int_from_pandas(index)

    y_pred = {}
    for degree in [1, 2]:
        forecaster = PolynomialTrendForecaster(degree=degree, with_intercept=False)
        forecaster.fit(y)
        y_pred[degree] = forecaster.predict(fh).to_numpy()

        linreg = forecaster.regressor_.steps[-1][1]
        assert linreg.intercept_ == 0
        assert linreg.coef_.shape == (degree,)
        np.testing.assert_allclose(
            forecaster.regressor_.predict(X_orig), y_pred[degree], rtol=1e-8
        )

    # the quadratic fit, in original units, is close to the stable reference fit
    expected = get_expected_polynomial_coefs(y, 2, with_intercept=False)[::-1]
    expected_pred = (X_orig ** np.arange(1, 3)) @ expected
    np.testing.assert_allclose(y_pred[2], expected_pred, rtol=1e-3)


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_polynomial_trend_predict_interval_large_time_stamps():
    """Test prediction intervals are stable for large time stamps, see #8686."""
    index = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    n = len(index)
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(0, 1, n), index=index)

    forecaster = PolynomialTrendForecaster(degree=3, prediction_intervals=True)
    forecaster.fit(y)
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    pred_var = forecaster.predict_var(fh=fh)

    # residual variance is ~1, and the leverage term is small, so var ~ 1
    assert np.isfinite(pred_var.to_numpy()).all()
    np.testing.assert_allclose(pred_var.to_numpy(), 1.0, rtol=0.1)
