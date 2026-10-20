# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Tests for TimestralForecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timestral import TimestralForecaster
from sktime.tests.test_switch import run_test_for_class


def test_timestral_license_rejected_by_default():
    """Verify license_accepted=False raises a clear ValueError."""
    forecaster = TimestralForecaster(license_accepted=False)
    with pytest.raises(ValueError, match="license_accepted"):
        forecaster._check_license()


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_univariate_point_forecast():
    """Univariate point forecast on load_airline using Timestral-5M."""
    y_train = load_airline().iloc[:-12]
    fh = [1, 2, 3]

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        device="cpu",
        license_accepted=True,
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=fh)

    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred) == len(fh)
    assert y_pred.shape[1] == 1
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_multivariate_with_covariates():
    """Multivariate and mixed past/known covariates forecast."""
    rng = np.random.default_rng(42)
    n, horizon = 32, 4
    ta = rng.normal(size=n)
    tb = rng.normal(size=n)
    po = rng.normal(size=n)
    fk = rng.normal(size=n)
    fkf = rng.normal(size=horizon)

    y = pd.DataFrame({"target_a": ta, "target_b": tb})
    X = pd.DataFrame({"past_only": po, "future_known": fk})
    X_future = pd.DataFrame({"future_known": fkf})

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        device="cpu",
        license_accepted=True,
    )
    forecaster.fit(y, X=X)
    fh = list(range(1, horizon + 1))
    y_pred = forecaster.predict(fh=fh, X=X_future)

    assert isinstance(y_pred, pd.DataFrame)
    assert list(y_pred.columns) == ["target_a", "target_b"]
    assert len(y_pred) == horizon
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_categorical_covariates():
    """Test categorical covariates encoding and forecasting."""
    n, horizon = 30, 3
    rng = np.random.default_rng(7)
    y = pd.DataFrame({"target": rng.normal(size=n)})
    X = pd.DataFrame(
        {
            "cat_known": pd.Categorical(["A", "B", "C"] * 10),
            "num_past": rng.normal(size=n),
        }
    )
    X_future = pd.DataFrame({"cat_known": pd.Categorical(["A", "B", "C"])})

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        device="cpu",
        license_accepted=True,
    )
    forecaster.fit(y, X=X)
    fh = list(range(1, horizon + 1))
    y_pred = forecaster.predict(fh=fh, X=X_future)

    assert isinstance(y_pred, pd.DataFrame)
    assert len(y_pred) == horizon
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_predict_quantiles():
    """Verify quantile prediction output structure and ordering."""
    y_train = load_airline().iloc[:-12]
    fh = [1, 2]
    alpha = [0.1, 0.5, 0.9]

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        device="cpu",
        license_accepted=True,
    )
    forecaster.fit(y_train)
    quantiles = forecaster.predict_quantiles(fh=fh, alpha=alpha)

    assert isinstance(quantiles, pd.DataFrame)
    assert isinstance(quantiles.columns, pd.MultiIndex)
    assert len(quantiles) == len(fh)
    # Check monotonicity of predicted quantiles
    q_vals = quantiles.to_numpy()
    assert (q_vals[:, 0] <= q_vals[:, 1]).all()
    assert (q_vals[:, 1] <= q_vals[:, 2]).all()


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_predict_proba_matches_quantiles():
    """predict_proba().quantile agrees with predict_quantiles."""
    y_train = load_airline().iloc[:-12]
    fh = [1, 2, 3]
    alpha = [0.1, 0.25, 0.5, 0.9]

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        device="cpu",
        license_accepted=True,
    )
    forecaster.fit(y_train)

    q_direct = forecaster.predict_quantiles(fh=fh, alpha=alpha)
    q_proba = forecaster.predict_proba(fh=fh).quantile(alpha=alpha)

    np.testing.assert_allclose(
        q_direct.to_numpy(), q_proba.to_numpy(), rtol=1e-5, atol=1e-4
    )
