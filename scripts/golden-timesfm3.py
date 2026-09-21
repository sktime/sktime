"""Generate golden reference outputs for ``TimesFM3Forecaster`` tests.

The reference forecasts in ``sktime/forecasting/tests/test_timesfm3.py`` are
produced by calling the upstream ``timesfm3`` package directly (no sktime
estimator), on the same data and parameters the tests use. This mirrors the
approach in ``scripts/golden-chronos.py`` and issue #10672.

Run this on a machine with ``timesfm[torch]>=3.0.0,<4.0.0`` installed and access
to the ``google/timesfm-3.0-pytorch`` checkpoint::

    python scripts/golden-timesfm3.py

Then paste the printed arrays into the ``EXPECTED_AIRLINE`` and ``EXPECTED_COV``
constants in the test module. Inference runs on CPU for reproducibility; the
array construction below is byte-for-byte what ``TimesFM3Forecaster`` feeds
upstream (target as ``(n_targets, n)``, covariates split into past-only and
past-and-future).
"""

import numpy as np

from sktime.datasets import load_airline  # dataset only, not the estimator


def _load_upstream():
    from timesfm3 import ModelConfig
    from timesfm3 import TimesFM3Forecaster as Upstream

    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=4,
        device="cpu",
    )
    return Upstream(config=config)


def airline_case(model):
    """Univariate point forecast on ``load_airline`` (fh = 1, 2, 3)."""
    y_train = load_airline().iloc[:-12]
    context = y_train.to_numpy().reshape(1, -1).astype(np.float32)
    out = model.predict(
        context=context, horizon=3, return_quantiles=True, sort_quantiles=True
    )
    head = np.asarray(out.forecast).ravel()[:3]
    print("EXPECTED_AIRLINE =", np.round(head, 5).tolist())


def covariate_case(model):
    """Multivariate + mixed covariate forecast (first two horizon steps)."""
    rng = np.random.default_rng(0)
    n, horizon = 32, 4
    # draw order must match the test module exactly
    ta = rng.normal(size=n)
    tb = rng.normal(size=n)
    po = rng.normal(size=n)
    fk = rng.normal(size=n)
    fkf = rng.normal(size=horizon)

    target = np.stack([ta, tb]).astype(np.float32)  # (2, n)
    past_only = po.reshape(1, -1).astype(np.float32)  # (1, n)
    past_future = np.concatenate([fk, fkf]).reshape(1, -1).astype(np.float32)

    out = model.predict(
        context=target,
        horizon=horizon,
        past_only_covariates=past_only,
        past_future_covariates=past_future,
        return_quantiles=True,
        sort_quantiles=True,
    )
    head = np.asarray(out.forecast).T[:2]  # (2 steps, 2 targets)
    print("EXPECTED_COV =", np.round(head, 5).tolist())


def main():
    """Print all golden reference arrays to paste into the test module."""
    model = _load_upstream()
    airline_case(model)
    covariate_case(model)


if __name__ == "__main__":
    main()
