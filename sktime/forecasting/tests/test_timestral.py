# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Golden-output regression tests for the Timestral forecaster.

Golden values were generated using the upstream mistralai/timestral standalone inference
pipeline with `mistralai/Timestral-5M` on CPU:

    # pip install git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
    import numpy as np
    from sktime.datasets import load_airline
    from timestral.forecaster import TimestralForecaster

    REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"

    # 1. Airline univariate reference:
    y = load_airline().iloc[:-12].values.reshape(1, 1, -1).astype(np.float32)
    m = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-5M",
        revision=REVISION,
        device="cpu",
    )
    pred = m.forecast(y, prediction_length=3)
    EXPECTED_AIRLINE = pred.median[0, 0].tolist()
    EXPECTED_AIRLINE_QUANTILES = {
        alpha: pred.quantile(alpha)[0, 0].tolist()
        for alpha in [0.1, 0.5, 0.9]
    }

    # 2. Covariates multivariate reference:
    rng = np.random.default_rng(0)
    n, horizon = 32, 4
    ta, tb = rng.normal(size=n), rng.normal(size=n)
    po = rng.normal(size=n)
    fk, fkf = rng.normal(size=n), rng.normal(size=horizon)

    target = np.stack([ta, tb], axis=0).astype(np.float32)[np.newaxis, ...]
    num_past = po.astype(np.float32).reshape(1, 1, n)
    num_known = np.concatenate([fk, fkf]).astype(np.float32).reshape(1, 1, n + horizon)

    pred_cov = m.forecast(
        context=target,
        prediction_length=horizon,
        past_covariates=num_past,
        known_covariates=num_known,
    )
    EXPECTED_COV = pred_cov.median[0].T.tolist()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timestral import TimestralForecaster
from sktime.tests.test_switch import run_test_for_class

# Golden references generated against `mistralai/Timestral-5M` on CPU
HF_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"
EXPECTED_AIRLINE = [414.82183837890625, 397.8201904296875, 452.1973876953125]
EXPECTED_AIRLINE_QUANTILES = {
    0.1: [400.60736083984375, 380.8467102050781, 429.85101318359375],
    0.5: [414.82183837890625, 397.8201904296875, 452.1973876953125],
    0.9: [429.47552490234375, 415.36260986328125, 474.31512451171875],
}
EXPECTED_COV = [
    [-0.3252118229866028, 0.12906599044799805],
    [-0.24099311232566833, 0.1534138321876526],
    [-0.2529636025428772, 0.16127654910087585],
    [-0.2486463487148285, 0.16425420343875885],
]


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_airline_matches_source_reference():
    """Univariate point forecast matches reference on ``load_airline``."""
    y_train = load_airline().iloc[:-12]
    fh = [1, 2, 3]

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        revision=HF_REVISION,
        device="cpu",
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=fh)

    np.testing.assert_allclose(
        y_pred.to_numpy().ravel(),
        np.asarray(EXPECTED_AIRLINE, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )

    # Quantile outputs match golden references
    alphas = list(EXPECTED_AIRLINE_QUANTILES.keys())
    q_pred = forecaster.predict_quantiles(fh=fh, alpha=alphas)
    for alpha in alphas:
        np.testing.assert_allclose(
            q_pred[y_train.name, alpha].to_numpy(),
            np.asarray(EXPECTED_AIRLINE_QUANTILES[alpha], dtype=np.float32),
            rtol=1e-5,
            atol=1e-4,
        )


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_covariates_match_source_reference():
    """Multivariate + mixed covariate forecast matches the reference.

    The covariate split is inferred: ``past_only`` is supplied only in fit-time
    ``X`` (past-only), while ``future_known`` is in both fit-time and predict-time
    ``X`` (past-and-future).
    """
    rng = np.random.default_rng(0)
    n, horizon = 32, 4
    ta = rng.normal(size=n)
    tb = rng.normal(size=n)
    po = rng.normal(size=n)
    fk = rng.normal(size=n)
    fkf = rng.normal(size=horizon)

    y = pd.DataFrame({"target_a": ta, "target_b": tb})
    X = pd.DataFrame({"past_only": po, "future_known": fk})
    X_future = pd.DataFrame(
        {"future_known": fkf},
        index=range(n, n + horizon),
    )

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        revision=HF_REVISION,
        device="cpu",
    )
    forecaster.fit(y, X=X)
    y_pred = forecaster.predict(fh=list(range(1, horizon + 1)), X=X_future)

    np.testing.assert_allclose(
        y_pred.to_numpy(),
        np.asarray(EXPECTED_COV, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_encode_categorical():
    """_encode_categorical maps labels consistently and handles missing values."""
    forecaster = TimestralForecaster()

    # Fit-only series
    s_fit = pd.Series(["cat", "dog", None, "cat"])
    codes = forecaster._encode_categorical(s_fit)
    assert isinstance(codes, np.ndarray)
    assert codes.dtype == np.float32
    assert codes.shape == (4,)
    assert codes[0] == codes[3]
    assert codes[0] != codes[1]
    assert np.isnan(codes[2])

    # Fit + future series sharing category mapping
    s1 = pd.Series(["a", "b", "a"])
    s2 = pd.Series(["b", "c", None])
    c1, c2 = forecaster._encode_categorical(s1, s2)
    assert isinstance(c1, np.ndarray) and isinstance(c2, np.ndarray)
    assert c1.shape == (3,) and c2.shape == (3,)
    assert c1[1] == c2[0]  # category 'b' matches across fit and future
    assert np.isnan(c2[2])


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_categorical_covariates_predict():
    """Predict succeeds with past-only and future-known categorical covariates."""
    rng = np.random.default_rng(42)
    n, horizon = 32, 4
    y = pd.Series(rng.normal(size=n), name="target")

    X = pd.DataFrame(
        {
            "cat_past": ["type_a" if i % 2 == 0 else "type_b" for i in range(n)],
            "cat_known": ["state_1" if i % 3 == 0 else "state_2" for i in range(n)],
        }
    )
    X_future = pd.DataFrame(
        {
            "cat_known": ["state_1", "state_2", "state_1", "state_2"],
        },
        index=range(n, n + horizon),
    )

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        revision=HF_REVISION,
        device="cpu",
    )
    forecaster.fit(y, X=X)
    y_pred = forecaster.predict(fh=list(range(1, horizon + 1)), X=X_future)

    assert y_pred.shape == (horizon,)
    assert not y_pred.isna().any()


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_predict_proba_matches_quantiles():
    """predict_proba().quantile agrees with predict_quantiles, including off-grid."""
    y_train = load_airline().iloc[:-12]
    fh = [1, 2, 3]
    # Includes on-grid and off-grid alphas to exercise linear interpolation
    alpha = [0.05, 0.1, 0.33, 0.5, 0.72, 0.9, 0.95]

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        revision=HF_REVISION,
        device="cpu",
    )
    forecaster.fit(y_train)

    q_direct = forecaster.predict_quantiles(fh=fh, alpha=alpha)
    q_proba = forecaster.predict_proba(fh=fh).quantile(alpha=alpha)

    np.testing.assert_allclose(
        q_direct.to_numpy(), q_proba.to_numpy(), rtol=1e-5, atol=1e-4
    )


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_multi_patch_horizon():
    """Forecast across multiple patches (>32 steps) without errors."""
    y_train = load_airline().iloc[:-40]
    horizon = 35
    fh = list(range(1, horizon + 1))

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        revision=HF_REVISION,
        device="cpu",
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=fh)

    assert len(y_pred) == horizon
    assert not y_pred.isna().any().any()


@pytest.mark.skipif(
    not run_test_for_class(TimestralForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_timestral_max_context_truncation():
    """Test that max_context properly truncates context to the most recent window."""
    y_full = load_airline().iloc[:-12]
    fh = [1, 2, 3]

    forecaster_capped = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        revision=HF_REVISION,
        device="cpu",
        max_context=32,
    )
    forecaster_capped.fit(y_full)
    pred_capped = forecaster_capped.predict(fh=fh)

    forecaster_truncated = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        revision=HF_REVISION,
        device="cpu",
        max_context=32,
    )
    forecaster_truncated.fit(y_full.iloc[-32:])
    pred_truncated = forecaster_truncated.predict(fh=fh)

    np.testing.assert_allclose(
        pred_capped.to_numpy(),
        pred_truncated.to_numpy(),
        rtol=1e-5,
        atol=1e-4,
    )
