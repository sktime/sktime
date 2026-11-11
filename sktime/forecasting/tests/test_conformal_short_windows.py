# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for insufficient conformal calibration windows (#3700)."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor

from sktime.exceptions import InsufficientDataError
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.conformal import ConformalIntervals
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _series(index_type="range", n=40):
    index = {
        "range": pd.RangeIndex(n),
        "period": pd.period_range("2000-01", periods=n, freq="M"),
        "datetime": pd.date_range("2000-01-01", periods=n, freq="D"),
    }[index_type]
    return pd.Series(np.arange(n, dtype=float) + np.sin(np.arange(n)), index=index)


def _forecaster(**kwargs):
    reducer = make_reduction(DummyRegressor(), window_length=12, strategy="recursive")
    return ConformalIntervals(reducer, method="conformal", **kwargs)


@pytest.mark.parametrize("index_type", ["range", "period", "datetime"])
@pytest.mark.parametrize("early_fh", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_short_calibration_windows(index_type, early_fh, n_jobs):
    """Skip short fits and calibrate from the same valid rows as an explicit window."""
    y = _series(index_type)
    X = y.to_frame("exogenous")
    X_predict = _series(index_type, n=43).to_frame("exogenous")
    fh = [1, 2, 3]
    f = _forecaster(n_jobs=n_jobs)
    with pytest.warns(UserWarning, match="skipped a calibration window"):
        f.fit(y, X=X, fh=fh if early_fh else None)
        result = f.predict_interval(fh=fh, X=X_predict)

    # A 12-lag recursive reducer first has a training pair at length 13.
    matrix = f.residuals_matrix_ if early_fh else None
    if matrix is not None:
        assert matrix.loc[y.index[10:13]].isna().all().all()
        assert matrix.loc[y.index[13:]].notna().any().any()
    skipped = [r for r in f.forecasters_ if r["forecaster"] is None]
    assert len(skipped) == 3

    baseline = _forecaster(initial_window=13).fit(y, X=X, fh=fh)
    expected = baseline.predict_interval(X=X_predict)
    assert_frame_equal(result, expected)
    assert np.isfinite(result.to_numpy(dtype=float)).all()
    assert (result.iloc[:, 1] > result.iloc[:, 0]).all()


def test_no_residuals_after_skipping():
    """Do not manufacture zero-width intervals for uncalibrated horizons."""
    f = _forecaster()
    with pytest.warns(UserWarning, match="skipped a calibration window"):
        f.fit(_series(), fh=[30])
    with pytest.raises(ValueError, match="no valid calibration residuals.*horizon 30"):
        f.predict_interval()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_extend_skipped_calibration_windows(n_jobs):
    """Cached missing rows stay missing when the calibration matrix is extended."""
    y = _series(n=45)
    f = _forecaster(n_jobs=n_jobs)
    with pytest.warns(UserWarning, match="skipped a calibration window"):
        f.fit(y.iloc[:40], fh=[1, 2, 3])
    original = f.residuals_matrix_.copy()
    result = f._compute_sliding_residuals(
        y, None, f.forecaster, initial_window=None, sample_frac=None, update=True
    )
    assert result.loc[y.index[10:13]].isna().all().all()
    assert_frame_equal(result.loc[original.index, original.columns], original)
    assert result.iloc[-5:].notna().any().any()


class _FailingForecaster(BaseForecaster):
    """Raise unrelated errors only on calibration windows."""

    _tags = {"y_inner_mtype": "pd.Series", "requires-fh-in-fit": False}

    def __init__(self, error="fit_value"):
        self.error = error
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        if len(y) == 10:
            if self.error == "fit_value":
                raise ValueError("unrelated fit error")
            if self.error == "fit_runtime":
                raise RuntimeError("unrelated fit error")
        self.n_ = len(y)
        return self

    def _predict(self, fh, X=None):
        if self.error == "predict_value":
            raise ValueError("unrelated prediction error")
        return pd.Series(0.0, index=fh.to_absolute_index(self.cutoff))


@pytest.mark.parametrize(
    "error,exception,match",
    [
        ("fit_value", ValueError, "unrelated fit error"),
        ("fit_runtime", RuntimeError, "unrelated fit error"),
        ("predict_value", ValueError, "unrelated prediction error"),
    ],
)
def test_unrelated_errors_propagate(error, exception, match):
    """Only the explicit insufficient-data exception is handled."""
    f = ConformalIntervals(_FailingForecaster(error=error))
    with pytest.raises(exception, match=match):
        f.fit(_series(), fh=[1, 2, 3])


def test_insufficient_full_training_data():
    """The full-data fit must succeed before calibration can skip any rows."""
    with pytest.raises(InsufficientDataError):
        _forecaster().fit(_series(n=10), fh=[1, 2, 3])
