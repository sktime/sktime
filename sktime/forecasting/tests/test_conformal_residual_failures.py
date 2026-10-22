"""Regression tests for residual-forecast failures in ConformalIntervals."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.conformal import ConformalIntervals
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(ConformalIntervals),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _raise_index_error(*args, **kwargs):
    """Simulate a forecaster that cannot predict residuals for a short window."""
    raise IndexError("insufficient history")


@pytest.mark.parametrize("index_kind", ["range", "datetime"])
def test_failed_residual_prediction_leaves_missing_calibration(monkeypatch, index_kind):
    """A failed calibration row should contain NaNs, not raise UnboundLocalError."""
    if index_kind == "datetime":
        index = pd.date_range("2024-01-01", periods=9, freq="D")
    else:
        index = pd.RangeIndex(9)
    y = pd.Series(np.arange(len(index), dtype=float), index=index)
    monkeypatch.setattr(NaiveForecaster, "predict_residuals", _raise_index_error)

    model = ConformalIntervals(NaiveForecaster(), initial_window=3, n_jobs=1)
    model.fit(y, fh=[1])

    assert model.residuals_matrix_.shape == (6, 6)
    assert model.residuals_matrix_.isna().all().all()


@pytest.mark.parametrize("method", ["empirical", "empirical_residual", "conformal"])
def test_predict_interval_rejects_completely_missing_calibration(monkeypatch, method):
    """Prediction intervals must not substitute zeros for failed calibration."""
    y = pd.Series(np.arange(9, dtype=float), index=pd.RangeIndex(9))
    monkeypatch.setattr(NaiveForecaster, "predict_residuals", _raise_index_error)

    model = ConformalIntervals(
        NaiveForecaster(), method=method, initial_window=3, n_jobs=1
    )
    model.fit(y, fh=[1])
    assert model.residuals_matrix_.isna().all().all()

    with pytest.raises(ValueError, match="no valid calibration residuals"):
        model.predict_interval(fh=[1], coverage=[0.8])


def test_failed_residual_update_preserves_previous_calibration(monkeypatch):
    """An update failure must not erase residuals calibrated during fit."""
    index = pd.period_range("2024-01", periods=9, freq="M")
    y = pd.Series(np.arange(9, dtype=float) ** 2, index=index)
    model = ConformalIntervals(NaiveForecaster(), initial_window=3, n_jobs=1)
    model.fit(y, fh=[1])
    old_residuals = model.residuals_matrix_.copy()
    assert old_residuals.notna().any().any()

    extended_index = pd.period_range("2024-01", periods=11, freq="M")
    extended_y = pd.Series(np.arange(11, dtype=float) ** 2, index=extended_index)
    monkeypatch.setattr(NaiveForecaster, "predict_residuals", _raise_index_error)
    updated = model._compute_sliding_residuals(
        y=extended_y,
        X=None,
        forecaster=NaiveForecaster(),
        initial_window=3,
        sample_frac=None,
        update=True,
    )

    pd.testing.assert_frame_equal(
        updated.loc[old_residuals.index, old_residuals.columns],
        old_residuals,
    )
    added = extended_y.index.difference(y.index)
    assert updated.loc[:, added].isna().all().all()
