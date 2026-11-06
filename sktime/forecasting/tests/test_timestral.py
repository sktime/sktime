# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Golden-output regression tests for the Timestral forecaster.

Golden values were generated using the upstream mistralai/timestral standalone inference
pipeline with `mistralai/Timestral-5M` on CPU:

    import numpy as np
    from sktime.datasets import load_airline
    from sktime.libs.timestral.forecaster import TimestralForecaster

    # 1. Airline univariate reference:
    y = load_airline().iloc[:-12].values.reshape(1, 1, -1).astype(np.float32)
    m = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device="cpu")
    pred = m.forecast(y, prediction_length=3)
    EXPECTED_AIRLINE = pred.median[0, 0].tolist()

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
    EXPECTED_COV = pred_cov.median[0, :, :2].T.tolist()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timestral import TimestralForecaster
from sktime.tests.test_switch import run_test_for_class

# Golden references generated against `mistralai/Timestral-5M` on CPU
EXPECTED_AIRLINE = [414.82183837890625, 397.8201904296875, 452.1973876953125]
EXPECTED_COV = [
    [-0.3252118229866028, 0.12906599044799805],
    [-0.24099311232566833, 0.1534138321876526],
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
    X_future = pd.DataFrame({"future_known": fkf})

    forecaster = TimestralForecaster(
        model_path="mistralai/Timestral-5M",
        device="cpu",
    )
    forecaster.fit(y, X=X)
    y_pred = forecaster.predict(fh=list(range(1, horizon + 1)), X=X_future)

    np.testing.assert_allclose(
        y_pred.iloc[:2].to_numpy(),
        np.asarray(EXPECTED_COV, dtype=np.float32),
        rtol=1e-5,
        atol=1e-4,
    )


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
    )
    forecaster.fit(y_train)

    q_direct = forecaster.predict_quantiles(fh=fh, alpha=alpha)
    q_proba = forecaster.predict_proba(fh=fh).quantile(alpha=alpha)

    np.testing.assert_allclose(
        q_direct.to_numpy(), q_proba.to_numpy(), rtol=1e-5, atol=1e-4
    )
