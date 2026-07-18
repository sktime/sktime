"""Tests for TimesFM2Forecaster."""

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.timesfm2 import TimesFM2Forecaster


class _StubTimesFM2Model:
    """Deterministic stand-in for the transformers TimesFM prediction model.

    Emits full predictions with value ``step + 0.5`` in the point forecast
    column, and value ``step + q`` in the column of quantile level ``q``,
    for horizon steps 1, 2, ..., ``horizon_length``. The quantile columns
    are monotone in the level, as for a trained TimesFM model.
    """

    def __init__(self, quantiles, horizon_length):
        import torch

        self.config = type("_Config", (), {})()
        self.config.quantiles = quantiles
        self.config.horizon_length = horizon_length
        self.dtype = torch.float32
        self.device = "cpu"

        steps = torch.arange(1, horizon_length + 1, dtype=torch.float32)
        cols = [steps + 0.5] + [steps + q for q in quantiles]
        self._full_predictions = torch.stack(cols, dim=-1).unsqueeze(0)

    def __call__(self, past_values, **kwargs):
        output = type("_Output", (), {})()
        output.full_predictions = self._full_predictions
        return output


@pytest.mark.skipif(
    not _check_estimator_deps(TimesFM2Forecaster, severity="none"),
    reason="run test only if softdeps are present",
)
def test_timesfm2_predict_proba_consistent_with_predict_quantiles():
    """predict_proba quantiles must agree with predict_quantiles, see #10566.

    Uses a deterministic stub model so the test is checkable against
    closed-form values without model downloads:

    * ``predict_quantiles`` at native levels returns the model's native
      quantile forecasts (guards against off-by-one into the point
      forecast column of ``full_predictions``)
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

    params = TimesFM2Forecaster.get_test_params()[0]
    forecaster = TimesFM2Forecaster(**params)
    forecaster.fit(y)
    # replace the fitted model with the deterministic stub
    forecaster.model_ = _StubTimesFM2Model(quantiles, horizon_length)

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
