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
    """Compute expected coefficients from polynomial regression."""
    t_ix = _get_X_numpy_int_from_pandas(y.index).reshape(-1)
    # The forecaster shifts time by the first stamp. Match that here.
    t_ix = t_ix - t_ix[0]
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


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_polynomial_trend_predict_uses_fit_origin():
    """A horizon after the training series stays on the origin stored at fit."""
    y = make_forecasting_problem()
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    forecaster.fit(y, fh=[1, 2, 3])
    y_pred = forecaster.predict().to_numpy().reshape(-1)

    coef = forecaster.regressor_.steps[-1][1].coef_
    t = _get_X_numpy_int_from_pandas(forecaster.fh.to_absolute_index(forecaster.cutoff))
    t = t.reshape(-1)
    on_fit_origin = coef[0] + coef[1] * (t - forecaster._trend_origin_)
    np.testing.assert_allclose(y_pred, on_fit_origin, rtol=1e-5)

    on_horizon_start = coef[0] + coef[1] * (t - t[0])
    assert not np.allclose(y_pred, on_horizon_start, rtol=1e-5)


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


def _stable_polynomial_coefs(y, degree, origin):
    """Least squares polynomial coefficients in time since ``origin``.

    Computed on the domain [-1, 1] and converted back, so that the reference
    is numerically stable also for long series and higher degrees.
    Coefficients are in increasing order of powers.
    """
    t = _get_X_numpy_int_from_pandas(y.index).reshape(-1).astype("float64") - origin
    poly = np.polynomial.Polynomial.fit(t, y.to_numpy(), degree)
    return poly.convert().coef


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("freq", ["D", "h"])
def test_polynomial_trend_long_series_higher_degree(freq):
    """Test numerical stability for long series and higher degree, see #8686.

    Even when time is measured from the first training time point, the
    polynomial features of degree 2 or higher are severely ill-conditioned for
    long series, e.g., several years of daily or hourly data. Degrees 2, 3 and 4
    then used to yield nearly the same, wrong, fit.
    """
    index = pd.date_range("2019-01-01", "2024-12-31", freq=freq)
    n = len(index)
    t = np.linspace(-1, 1, n)
    trend = 1000 + 500 * t + 200 * t**2 - 300 * t**3
    rng = np.random.default_rng(42)
    y = pd.Series(trend + rng.normal(0, 20, n), index=index)
    fh = ForecastingHorizon(index, is_relative=False)

    y_pred = {}
    for degree in [2, 3, 4]:
        forecaster = PolynomialTrendForecaster(degree=degree)
        forecaster.fit(y)
        y_pred[degree] = forecaster.predict(fh).to_numpy().reshape(-1)

        # coefficients are in units of time since the first training time point,
        # see docstring, and must agree with a numerically stable reference fit
        actual = forecaster.regressor_.steps[-1][1].coef_
        expected = _stable_polynomial_coefs(y, degree, forecaster._trend_origin_)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-12)

        # the fitted pipeline, applied to time since the origin, agrees with predict
        X_since_origin = _get_X_numpy_int_from_pandas(index) - forecaster._trend_origin_
        np.testing.assert_allclose(
            forecaster.regressor_.predict(X_since_origin), y_pred[degree], rtol=1e-8
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
def test_polynomial_trend_long_series_no_intercept():
    """Test fitting a long series with ``with_intercept=False``, see #8686.

    Without intercept, the polynomial must vanish at the first training time
    point, so time is scaled but not shifted further internally. The coefficients
    must still be in the documented units, and the fitted pipeline must agree
    with ``predict``.
    """
    index = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    n = len(index)
    t = np.linspace(0, 1, n)
    rng = np.random.default_rng(42)
    y = pd.Series(500 * t + 200 * t**2 + rng.normal(0, 20, n), index=index)
    fh = ForecastingHorizon(index, is_relative=False)

    for degree in [1, 2]:
        forecaster = PolynomialTrendForecaster(degree=degree, with_intercept=False)
        forecaster.fit(y)
        y_pred = forecaster.predict(fh).to_numpy().reshape(-1)

        linreg = forecaster.regressor_.steps[-1][1]
        assert linreg.intercept_ == 0
        assert linreg.coef_.shape == (degree,)
        X_since_origin = _get_X_numpy_int_from_pandas(index) - forecaster._trend_origin_
        np.testing.assert_allclose(
            forecaster.regressor_.predict(X_since_origin), y_pred, rtol=1e-8
        )

    # the quadratic fit without intercept is close to the true trend
    assert np.sqrt(np.mean((y_pred - (500 * t + 200 * t**2)) ** 2)) < 5


@pytest.mark.skipif(
    not run_test_for_class([PolynomialTrendForecaster, _get_X_numpy_int_from_pandas]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_polynomial_trend_predict_interval_long_series():
    """Test prediction intervals are stable for long series, see #8686."""
    index = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    n = len(index)
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(0, 1, n), index=index)

    forecaster = PolynomialTrendForecaster(degree=3, prediction_intervals=True)
    forecaster.fit(y)
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    pred_var = forecaster.predict_var(fh=fh)

    # residual variance is ~1, and the variance of a 3 step ahead forecast
    # of a long series is only slightly larger than the residual variance
    assert np.all(pred_var.to_numpy() > 0.9)
    assert np.all(pred_var.to_numpy() < 1.2)

    pred_int = forecaster.predict_interval(fh=fh, coverage=0.9)
    assert np.all(np.isfinite(pred_int.to_numpy()))
