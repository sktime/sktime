# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for DeepDynamicFactor forecaster."""

import pytest

from sktime.datasets import load_longley
from sktime.forecasting.deep_dynamic_factor import DeepDynamicFactor
from sktime.tests.test_switch import run_test_for_class

__author__ = ["minkeymouse"]

# load_longley() returns (y, X); X has GNPDEFL, GNP, UNEMP, ARMED, POP (16 rows)
_, _longley_X = load_longley()
y_df = _longley_X[["GNPDEFL", "GNP"]]
X_df = _longley_X[["UNEMP", "POP"]]
TRAIN_SIZE = 12
PRED_LEN = 3

# Fast params for tests (small encoder, few MCMC iters); matches get_test_params().
_TEST_PARAMS = {
    "encoder_size": (8, 2),
    "max_iter": 2,
    "n_mc_samples": 2,
    "window_size": 4,
    "random_state": 42,
}


def test_deep_dynamic_factor_invalid_decoder_type_raises():
    """Test that invalid decoder_type raises ValueError (no soft deps)."""
    with pytest.raises(ValueError, match="decoder_type must be 'linear' or 'mlp'"):
        DeepDynamicFactor(decoder_type="invalid", **_TEST_PARAMS)


def test_deep_dynamic_factor_invalid_encoder_size_raises():
    """Test that invalid encoder_size raises ValueError (no soft deps)."""
    params = {k: v for k, v in _TEST_PARAMS.items() if k != "encoder_size"}
    with pytest.raises(ValueError, match="encoder_size must be a non-empty"):
        DeepDynamicFactor(encoder_size=(), **params)
    with pytest.raises(ValueError, match="encoder_size must be a non-empty"):
        DeepDynamicFactor(encoder_size=(8, 0), **params)


def test_deep_dynamic_factor_invalid_numeric_params_raise():
    """Test that max_iter/n_mc_samples/window_size < 1 raise (no soft deps)."""
    params = {k: v for k, v in _TEST_PARAMS.items() if k != "max_iter"}
    with pytest.raises(ValueError, match="must be >= 1"):
        DeepDynamicFactor(max_iter=0, **params)


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch) are present",
)
def test_deep_dynamic_factor_fit_raises_when_y_shorter_than_window_size():
    """Test that fit raises when len(y) < window_size."""
    # _TEST_PARAMS has window_size=4
    y_short = y_df.iloc[:2]
    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    with pytest.raises(ValueError, match="len\\(y\\) >= window_size"):
        forecaster.fit(y_short)


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch) are present",
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
    reason="run test only if softdeps (torch) are present",
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
    reason="run test only if softdeps (torch) are present",
)
def test_deep_dynamic_factor_get_fitted_params():
    """Test that fitted params include factors and state-space matrices."""
    y_train = y_df.iloc[:TRAIN_SIZE]
    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    forecaster.fit(y_train)
    params = forecaster.get_fitted_params()
    assert "factors" in params
    assert "model" in params
    assert params["factors"] is not None


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch) are present",
)
def test_deep_dynamic_factor_predict_empty_fh():
    """Test that empty fh raises (sktime validation forbids empty fh)."""
    y_train = y_df.iloc[:TRAIN_SIZE]
    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    forecaster.fit(y_train)
    from sktime.forecasting.base import ForecastingHorizon

    fh_empty = ForecastingHorizon([])
    with pytest.raises(ValueError, match="fh.*must not be empty"):
        forecaster.predict(fh=fh_empty)


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch) are present",
)
def test_deep_dynamic_factor_update():
    """Test that update() extends factors and predict still works."""
    y_train = y_df.iloc[:TRAIN_SIZE]
    y_new = y_df.iloc[TRAIN_SIZE : TRAIN_SIZE + 2]
    X_new = X_df.iloc[TRAIN_SIZE : TRAIN_SIZE + 2]
    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    forecaster.fit(y_train)
    forecaster.update(y_new, X=X_new, update_params=True)
    y_pred = forecaster.predict(fh=[1])
    assert y_pred.shape[0] == 1
    assert list(y_pred.columns) == list(y_train.columns)


@pytest.mark.skipif(
    not run_test_for_class(DeepDynamicFactor),
    reason="run test only if softdeps (torch) are present",
)
def test_deep_dynamic_factor_single_column_y():
    """Test fit/predict with univariate (single-column) y."""
    y_single = y_df[["GNP"]].iloc[:TRAIN_SIZE]
    forecaster = DeepDynamicFactor(**_TEST_PARAMS)
    forecaster.fit(y_single)
    y_pred = forecaster.predict(fh=[1, 2])
    assert y_pred.shape == (2, 1)
    assert list(y_pred.columns) == ["GNP"]
    assert not y_pred.isna().all().all()


def test_deep_dynamic_factor_get_test_params():
    """Test get_test_params returns list of dicts with expected keys (no soft deps)."""
    params_list = DeepDynamicFactor.get_test_params()
    assert isinstance(params_list, list)
    assert len(params_list) >= 1
    for p in params_list:
        assert isinstance(p, dict)
        assert "encoder_size" in p
        assert "random_state" in p
    # When torch is available, params are usable as constructor kwargs
    if run_test_for_class(DeepDynamicFactor):
        forecaster = DeepDynamicFactor(**params_list[0])
        assert forecaster.encoder_size == params_list[0]["encoder_size"]
