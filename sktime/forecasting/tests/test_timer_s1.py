"""Tests for TimerS1Forecaster."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.timer_s1 import TimerS1Forecaster


class _StubTimerS1Model:
    """Deterministic stand-in for Timer-S1 generate().

    Emits value ``step + q`` for horizon steps 1..horizon_length at each
    native quantile level ``q``. Output layout matches
    ``TimerS1Forecaster._predict_quantiles``: shape
    ``(n_quantiles, horizon_length)`` before the forecaster's ``.T``.
    """

    def __init__(self, quantiles, horizon_length):
        import torch

        self.config = type("_Config", (), {})()
        self.config.quantiles = quantiles
        self.dtype = torch.float32
        self.device = "cpu"

        steps = torch.arange(1, horizon_length + 1, dtype=torch.float32)
        # generate returns (batch=1, n_quantiles, horizon) after squeeze(0) path:
        # forecaster does output.squeeze(0) then .T → (horizon, n_quantiles)
        # so raw generate output should be (1, n_quantiles, horizon)
        cols = torch.stack([steps + q for q in quantiles], dim=0)
        self._output = cols.unsqueeze(0)

    def generate(self, past_values, max_new_tokens=None, **kwargs):
        return self._output


@pytest.mark.skipif(
    not _check_estimator_deps(TimerS1Forecaster, severity="none"),
    reason="run test only if softdeps are present",
)
def test_timer_s1_predict_proba_consistent_with_predict_quantiles():
    """predict_proba quantiles must agree with predict_quantiles, see #10566.

    Uses a deterministic stub model so the test is checkable against
    closed-form values without model downloads:

    * ``predict_quantiles`` at native levels returns the model's native
      quantile forecasts
    * ``predict_proba`` returns a ``HistogramQPD``, not the Normal fallback
    * quantiles derived from ``predict_proba`` match ``predict_quantiles``
      at native levels, interpolate linearly between native levels, and
      clamp to the boundary quantiles outside the native grid
    """
    pytest.importorskip("skpro")
    from skpro.distributions import HistogramQPD

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    horizon_length = 6

    y = pd.Series(np.sin(np.linspace(0, 10, 30)), name="y")

    params = TimerS1Forecaster.get_test_params()[0]
    # keep config quantiles aligned with the stub
    params = dict(params)
    params["config"] = dict(params["config"])
    params["config"]["quantiles"] = quantiles

    forecaster = TimerS1Forecaster(**params)
    with pytest.warns(UserWarning, match="random weights"):
        forecaster.fit(y)
    forecaster.model_ = _StubTimerS1Model(quantiles, horizon_length)

    fh = [1, 3, 5]
    steps = np.array(fh, dtype=float)
    alpha = [0.1, 0.5, 0.9]

    q_direct = forecaster.predict_quantiles(fh=fh, alpha=alpha)
    expected = np.column_stack([steps + a for a in alpha])
    np.testing.assert_allclose(q_direct.to_numpy(), expected)

    pred_dist = forecaster.predict_proba(fh=fh)
    assert isinstance(pred_dist, HistogramQPD)

    q_from_proba = pred_dist.quantile(alpha=alpha)
    np.testing.assert_allclose(q_direct.to_numpy(), q_from_proba.to_numpy(), rtol=1e-6)

    # interpolated level between native grid points 0.25 and 0.5
    q_interp = pred_dist.quantile(alpha=[0.375])
    np.testing.assert_allclose(q_interp.to_numpy().ravel(), steps + 0.375, rtol=1e-6)

    # levels outside the native grid clamp to the boundary quantiles
    q_clamped = pred_dist.quantile(alpha=[0.01, 0.99])
    np.testing.assert_allclose(q_clamped.to_numpy()[:, 0], steps + 0.1, rtol=1e-6)
    np.testing.assert_allclose(q_clamped.to_numpy()[:, 1], steps + 0.9, rtol=1e-6)
