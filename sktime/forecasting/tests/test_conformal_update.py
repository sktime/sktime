# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for incremental conformal calibration history (#11845)."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from sktime.forecasting.conformal import ConformalIntervals
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(24),
        pd.period_range("2020-01", periods=24, freq="M"),
        pd.date_range("2020-01-01", periods=24, freq="D"),
    ],
    ids=["range", "period", "datetime"],
)
@pytest.mark.parametrize("method", ["empirical", "conformal"])
@pytest.mark.parametrize("batch_size", [1, 3, 4])
@pytest.mark.parametrize("early_fh", [True, False])
def test_conformal_update_matches_full_history(index, method, batch_size, early_fh):
    """Batching and deferred horizons must not change the calibration data."""
    y = pd.Series(np.arange(24.0) ** 2, index=index, name="target")
    f = ConformalIntervals(NaiveForecaster(), method=method, initial_window=3)
    f.fit(y.iloc[:20], fh=[1, 2] if early_fh else None)

    for start in range(20, len(y), batch_size):
        f.update(y.iloc[start : start + batch_size])

    expected = f.clone().fit(y, fh=[1, 2])
    assert_series_equal(f.predict(fh=[1, 2]), expected.predict())
    assert_frame_equal(f.predict_interval(), expected.predict_interval())
    if early_fh:
        assert_frame_equal(f.residuals_matrix_, expected.residuals_matrix_)


@pytest.mark.parametrize("initial_window", [None, 0.2])
def test_conformal_update_recomputes_after_corrections(initial_window):
    """Overlapping updates replace observations used by old calibration fits."""
    y = pd.Series(np.arange(24.0) ** 2, name="target")
    f = ConformalIntervals(NaiveForecaster(), initial_window=initial_window)
    f.fit(y.iloc[:20], fh=[1, 2])
    corrected = y.iloc[15:20] + 100
    f.update(corrected)
    f.update(y.iloc[20:])

    y.loc[corrected.index] = corrected
    expected = f.clone().fit(y, fh=[1, 2])
    assert_frame_equal(f.predict_interval(), expected.predict_interval())


def test_conformal_update_preserves_calibration_without_parameter_updates():
    """Retain observations for later calibration without changing fitted residuals."""
    y = pd.Series(np.arange(24.0) ** 2, name="target")
    f = ConformalIntervals(NaiveForecaster(), initial_window=3)
    f.fit(y.iloc[:20], fh=[1, 2])
    residuals = f.residuals_matrix_.copy()
    widths = np.diff(f.predict_interval().to_numpy(dtype=float), axis=1)

    f.update(y.iloc[20:23], update_params=False)

    assert f.cutoff[0] == 22
    assert f.forecaster_.cutoff[0] == 22
    assert_frame_equal(f.residuals_matrix_, residuals)
    np.testing.assert_allclose(
        np.diff(f.predict_interval().to_numpy(dtype=float), axis=1), widths
    )

    f.update(y.iloc[23:])
    expected = f.clone().fit(y, fh=[1, 2])
    assert_frame_equal(f.predict_interval(), expected.predict_interval())


@pytest.mark.parametrize("early_fh", [True, False])
def test_conformal_update_retains_exogenous_history(early_fh):
    """Calibration receives matching accumulated target and exogenous observations."""
    from sklearn.linear_model import LinearRegression

    from sktime.forecasting.compose import YfromX

    X = pd.DataFrame({"x": np.arange(24.0)})
    y = (X["x"] ** 2).rename("target")
    f = ConformalIntervals(YfromX(LinearRegression()), initial_window=3)
    f.fit(y.iloc[:20], X=X.iloc[:20], fh=[1, 2] if early_fh else None)
    f.update(y.iloc[20:], X=X.iloc[20:])

    expected = f.clone().fit(y, X=X, fh=[1, 2])
    # The inner regressor does not implement parameter updates. Compare calibrated
    # offsets rather than point predictions from its unchanged fitted coefficients.
    future_X = pd.DataFrame({"x": [24.0, 25.0]}, index=[24, 25])
    actual = f.predict_interval(fh=[1, 2], X=future_X)
    expected_interval = expected.predict_interval(X=future_X)
    actual_offsets = (
        actual.to_numpy(dtype=float) - f.predict(X=future_X).to_numpy()[:, None]
    )
    expected_offsets = (
        expected_interval.to_numpy(dtype=float)
        - expected.predict(X=future_X).to_numpy()[:, None]
    )
    np.testing.assert_allclose(actual_offsets, expected_offsets, atol=1e-10)


@pytest.mark.parametrize("sample_frac", [None, 1.0])
def test_conformal_update_with_parallel_calibration(sample_frac):
    """Recalibration also works when worker-local forecasters are not cached."""
    y = pd.Series(np.arange(16.0) ** 2, name="target")
    f = ConformalIntervals(
        NaiveForecaster(), initial_window=3, n_jobs=2, sample_frac=sample_frac
    ).fit(y.iloc[:12], fh=[1, 2])
    f.update(y.iloc[12:])

    expected = f.clone().fit(y, fh=[1, 2])
    assert_frame_equal(f.predict_interval(), expected.predict_interval())
