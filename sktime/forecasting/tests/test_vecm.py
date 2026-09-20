"""Tests the VAR model."""

__author__ = ["thayeylolu", "AurumnPegasus"]
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.vecm import VECM
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(VECM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VECM_against_statsmodels():
    """Compares Sktime's and Statsmodel's VECM."""
    from statsmodels.tsa.api import VECM as _VECM

    pandas2 = _check_soft_dependencies("pandas>=2.0.0", severity="none")
    if pandas2:
        freq = "ME"
    else:
        freq = "M"

    index = pd.date_range(start="2005", end="2006-12", freq=freq)
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(23, 2)),
        columns=list("AB"),
        index=pd.PeriodIndex(index),
    )
    train, test = temporal_train_test_split(df)
    sktime_model = VECM()
    fh = ForecastingHorizon([1, 3, 4, 5, 7, 9])
    _ = sktime_model.fit(train)
    y_pred = sktime_model.predict(fh=fh)

    stats = _VECM(train)
    stats_fit = stats.fit()
    fh_int = fh.to_relative(train.index[-1])
    # lagged = stats_fit.k_ar
    y_pred_stats = stats_fit.predict(steps=fh_int[-1])
    new_arr = []
    for i in fh_int:
        new_arr.append(y_pred_stats[i - 1])
    # print("predicted: \n")
    # print(y_pred)
    # print("actual: \n")
    # print(new_arr)
    assert_allclose(y_pred, new_arr)


@pytest.mark.skipif(
    not run_test_for_class(VECM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VECM_insample_prediction_with_custom_columns():
    """Test VECM in-sample prediction with non-default column names.

    Regression test for #6633: in-sample prediction should work with
    DataFrames that have non-default column names and datetime index,
    and match the residuals-based in-sample fit.
    """
    np.random.seed(42)
    data = np.random.randn(100, 2)
    index = pd.date_range(start="2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(data, columns=["a", "b"], index=index)

    y_train, y_test = temporal_train_test_split(df)

    model = VECM(k_ar_diff=2, coint_rank=1, deterministic="ci", seasons=0)
    model.fit(y_train)

    # In-sample forecast horizon (negative values)
    fh = ForecastingHorizon(range(-4, 1), is_relative=True)
    y_pred = model.predict(fh)

    # Check that prediction has correct shape and column names
    assert y_pred.shape[1] == 2
    assert list(y_pred.columns) == ["a", "b"]
    # Check that predictions are finite
    assert np.all(np.isfinite(y_pred.values))

    # in-sample predictions are y - resid, with resid aligned to the tail
    # of y_train; verify this explicitly against the fitted residuals
    resid = model._fitted_forecaster.resid
    offset = len(y_train) - len(resid)
    y_hat = y_train.values[offset:] - resid
    expected = y_hat[len(y_hat) - len(y_pred) :]
    assert_allclose(y_pred.values, expected)


@pytest.mark.skipif(
    not run_test_for_class(VECM),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_VECM_mixed_horizon_prediction_with_custom_columns():
    """Test VECM prediction with mixed in-sample and out-sample horizon.

    Regression test for #6633: prediction with a horizon that contains
    both in-sample (fh <= 0) and out-sample (fh > 0) values should work
    with DataFrames that have non-default column names and datetime index.
    """
    np.random.seed(42)
    data = np.random.randn(100, 2)
    index = pd.date_range(start="2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(data, columns=["a", "b"], index=index)

    y_train, y_test = temporal_train_test_split(df)

    model = VECM(k_ar_diff=2, coint_rank=1, deterministic="ci", seasons=0)
    model.fit(y_train)

    fh = ForecastingHorizon(range(-2, 3), is_relative=True)
    y_pred = model.predict(fh)

    assert y_pred.shape == (5, 2)
    assert list(y_pred.columns) == ["a", "b"]
    assert np.all(np.isfinite(y_pred.values))
