"""Tests for FlowStateForecaster probabilistic forecasts."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.flowstate import FlowStateForecaster


class _ArrayOutput:
    """Minimal tensor-like wrapper used by the FlowState adapter."""

    def __init__(self, values):
        self._values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


def _make_forecaster():
    """Return a fitted FlowState forecaster backed by deterministic quantiles."""
    y = pd.DataFrame({"y": np.arange(10, dtype=float)}, index=pd.RangeIndex(10))
    forecaster = FlowStateForecaster()
    forecaster.model = SimpleNamespace(
        config=SimpleNamespace(quantiles=[0.1, 0.5, 0.9]), eval=lambda: None
    )

    def _run(pred_len):
        quantiles = np.empty((1, 3, pred_len, 1), dtype=float)
        for step in range(pred_len):
            quantiles[0, :, step, 0] = [step + 1.0, step + 2.0, step + 4.0]
        predictions = np.arange(2.0, pred_len + 2.0).reshape(1, pred_len, 1)
        return SimpleNamespace(
            prediction_outputs=_ArrayOutput(predictions),
            quantile_outputs=_ArrayOutput(quantiles),
        )

    with patch.object(forecaster, "_load_model", return_value=forecaster.model):
        with patch("sktime.forecasting.base._base._check_estimator_deps"):
            forecaster.fit(y)
    forecaster._run = _run
    return forecaster


@pytest.mark.skipif(
    not _check_soft_dependencies("skpro>=2.14", severity="none"),
    reason="skpro>=2.14 with HistogramQPD",
)
def test_predict_proba_uses_flowstate_quantile_grid():
    """predict_proba returns the same native and interpolated quantiles."""
    from skpro.distributions import HistogramQPD

    forecaster = _make_forecaster()
    fh = [1, 3]
    alpha = [0.01, 0.1, 0.3, 0.9, 0.99]

    pred_dist = forecaster.predict_proba(fh=fh)
    pred_quantiles = forecaster.predict_quantiles(fh=fh, alpha=alpha)

    assert isinstance(pred_dist, HistogramQPD)
    pd.testing.assert_frame_equal(
        pred_dist.quantile(alpha=alpha), pred_quantiles, check_dtype=False
    )
