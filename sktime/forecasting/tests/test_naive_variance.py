"""Value-level regression tests for NaiveVariance calibration."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster, NaiveVariance
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(NaiveVariance),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _rolling_errors(y, horizon, initial_window, strategy):
    """Compute errors directly from each training prefix and its h-step target."""
    errors = []
    for n_train in range(initial_window, len(y) - horizon + 1):
        prediction = y.iloc[n_train - 1]
        # NaiveForecaster falls back to the last value for a one-point window.
        if strategy == "drift" and n_train > 1:
            prediction += horizon * (prediction - y.iloc[0]) / (n_train - 1)
        errors.append(y.iloc[n_train + horizon - 1] - prediction)
    return np.asarray(errors)


@pytest.mark.parametrize("index_type", ["range", "period", "datetime"])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("fit_fh", [False, True])
@pytest.mark.parametrize("strategy", ["last", "drift"])
@pytest.mark.parametrize("initial_window", [1, 3])
def test_naive_variance_horizon_calibration(
    index_type, absolute, fit_fh, strategy, initial_window
):
    """Variances and covariances use errors at the requested, non-contiguous steps."""
    n = 12
    if index_type == "period":
        index = pd.period_range("2020-01", periods=n, freq="M")
    elif index_type == "datetime":
        index = pd.date_range("2020-01-01", periods=n, freq="D")
    else:
        index = pd.RangeIndex(20, 20 + n)
    y = pd.Series(np.arange(n, dtype=float) ** 2, index=index, name="y")
    relative_fh = ForecastingHorizon([1, 3, 5], freq=getattr(index, "freq", None))
    expected_index = relative_fh.to_absolute(index[-1]).to_pandas()
    fh = (
        ForecastingHorizon(expected_index, is_relative=False)
        if absolute
        else relative_fh
    )
    model = NaiveVariance(
        NaiveForecaster(strategy=strategy), initial_window=initial_window
    ).fit(y, fh=fh if fit_fh else None)

    errors = [_rolling_errors(y, h, initial_window, strategy) for h in relative_fh]
    expected_variance = [np.nanmean(error**2) for error in errors]
    expected_covariance = np.empty((len(errors), len(errors)))
    for i, left in enumerate(errors):
        for j, right in enumerate(errors):
            # Pair errors from the same training prefix, not the same target date.
            n_common = min(len(left), len(right))
            expected_covariance[i, j] = np.nanmean(left[:n_common] * right[:n_common])

    variance = model.predict_var(fh)
    covariance = model.predict_var(fh, cov=True)
    pd.testing.assert_index_equal(variance.index, expected_index)
    pd.testing.assert_index_equal(covariance.index, expected_index)
    pd.testing.assert_index_equal(covariance.columns, expected_index)
    np.testing.assert_allclose(variance.iloc[:, 0], expected_variance)
    np.testing.assert_allclose(covariance, expected_covariance)
    np.testing.assert_allclose(np.diag(covariance), variance.iloc[:, 0])

    alpha = [0.1, 0.5, 0.9]
    quantiles = model.predict_quantiles(fh, alpha=alpha)
    expected_quantiles = model.predict(fh).to_numpy()[:, None] + np.sqrt(
        expected_variance
    )[:, None] * norm.ppf(alpha)
    np.testing.assert_allclose(quantiles, expected_quantiles)


@pytest.mark.parametrize("fit_fh", [False, True])
def test_naive_variance_last_available_horizon(fit_fh):
    """The last calibratable step has one residual, rather than an empty diagonal."""
    y = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    initial_window = 2
    horizon = len(y) - initial_window
    model = NaiveVariance(NaiveForecaster(), initial_window=initial_window).fit(
        y, fh=[horizon] if fit_fh else None
    )
    expected = (y.iloc[-1] - y.iloc[initial_window - 1]) ** 2
    np.testing.assert_allclose(model.predict_var([horizon]), [[expected]])
    np.testing.assert_allclose(model.predict_var([horizon], cov=True), [[expected]])


def test_naive_variance_skipped_fitting_windows():
    """Covariance pairs the same origins even when early fits are unavailable."""
    y = pd.Series(np.arange(10, dtype=float) ** 2)
    horizons = [1, 3]
    model = NaiveVariance(NaiveForecaster(window_length=3)).fit(y, fh=horizons)
    errors = [_rolling_errors(y, h, 1, "last") for h in horizons]
    for error in errors:
        error[:2] = np.nan  # The first two prefixes cannot fit a length-3 window.
    expected_variance = [np.nanmean(error**2) for error in errors]
    expected_cross = np.nanmean(errors[0][: len(errors[1])] * errors[1])
    expected_covariance = [
        [expected_variance[0], expected_cross],
        [expected_cross, expected_variance[1]],
    ]
    np.testing.assert_allclose(
        model.predict_var(horizons).iloc[:, 0], expected_variance
    )
    np.testing.assert_allclose(
        model.predict_var(horizons, cov=True), expected_covariance
    )


@pytest.mark.parametrize("fit_fh", [False, True])
def test_naive_variance_updated_calibration(fit_fh):
    """Updating the training data keeps h-step variance and covariance aligned."""
    y = pd.Series(np.arange(10, dtype=float) ** 2)
    horizons = [1, 3]
    initial_window = 2
    model = NaiveVariance(NaiveForecaster(), initial_window=initial_window).fit(
        y.iloc[:7], fh=horizons if fit_fh else None
    )
    model.update(y.iloc[7:])
    errors = [_rolling_errors(y, h, initial_window, "last") for h in horizons]
    expected_variance = [np.mean(error**2) for error in errors]
    n_common = len(errors[1])
    expected_cross = np.mean(errors[0][:n_common] * errors[1])
    expected_covariance = [
        [expected_variance[0], expected_cross],
        [expected_cross, expected_variance[1]],
    ]
    np.testing.assert_allclose(
        model.predict_var(horizons).iloc[:, 0], expected_variance
    )
    np.testing.assert_allclose(
        model.predict_var(horizons, cov=True), expected_covariance
    )
