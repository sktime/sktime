# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for DeepDynamicFactor forecaster."""

import pytest

from sktime.datasets import load_longley
from sktime.forecasting.deep_dynamic_factor import DeepDynamicFactor
from sktime.tests.test_switch import run_test_for_class

__author__ = ["minkeymouse"]

y_df = load_longley()[["GNPDEFL", "GNP"]].iloc[:30]
X_df = load_longley()[["UNEMP", "POP"]].iloc[:30]
TRAIN_SIZE = 24
PRED_LEN = 3

# Fast params for tests (small encoder, few MCMC iters); matches get_test_params().
_TEST_PARAMS = {
    "encoder_size": (8, 2),
    "max_iter": 2,
    "n_mc_samples": 2,
    "window_size": 4,
    "seed": 42,
}


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch, dfm_python) are present",
)
def test_deep_dynamic_factor_fit_predict_no_X():
    """Test DeepDynamicFactor fit and predict without exogenous variables."""
    y_train = y_df.iloc[:TRAIN_SIZE]
    fh = list(range(1, PRED_LEN + 1))
    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    forecaster.fit(y_train, fh=fh)
    y_pred = forecaster.predict(fh=fh)
    assert y_pred.shape[0] == PRED_LEN
    assert list(y_pred.columns) == list(y_train.columns)
    assert not y_pred.isna().all().all()


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch, dfm_python) are present",
)
def test_deep_dynamic_factor_fit_predict_with_X():
    """Test DeepDynamicFactor fit and predict with exogenous variables."""
    y_train = y_df.iloc[:TRAIN_SIZE]
    X_train = X_df.iloc[:TRAIN_SIZE]
    X_test = X_df.iloc[TRAIN_SIZE : TRAIN_SIZE + PRED_LEN]
    fh = list(range(1, PRED_LEN + 1))
    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    forecaster.fit(y_train, X=X_train, fh=fh)
    y_pred = forecaster.predict(fh=fh, X=X_test)
    assert y_pred.shape[0] == PRED_LEN
    assert list(y_pred.columns) == list(y_train.columns)
    assert not y_pred.isna().all().all()


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch, dfm_python) are present",
)
def test_deep_dynamic_factor_get_fitted_params():
    """Test that fitted params include factors and state-space matrices."""
    y_train = y_df.iloc[:TRAIN_SIZE]
    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    forecaster.fit(y_train)
    params = forecaster.get_fitted_params()
    assert "factors" in params
    assert "ddfm" in params
    assert params["factors"] is not None
