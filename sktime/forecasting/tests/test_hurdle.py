"""Tests for Hurdle estimator."""

import numpy as np
import pytest
from prophetverse.engine import MAPInferenceEngine, MCMCInferenceEngine

from sktime.datasets import load_PBS_dataset
from sktime.forecasting.hurdle_demand import HurdleDemandForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(HurdleDemandForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("family", ["poisson", "gamma-poisson"])
@pytest.mark.parametrize("time_varying", [True, False])
@pytest.mark.parametrize(
    "engine",
    [
        MCMCInferenceEngine(num_samples=5, num_warmup=50, num_chains=4),
        MAPInferenceEngine(num_steps=10, num_samples=10),
    ],
)
def test_hurdle_model(family: str, time_varying: bool, engine):
    """Test that Hurdle model can be instantiated and run with default parameters."""
    y = load_PBS_dataset()
    forecaster = HurdleDemandForecaster(
        family=family,
        time_varying_demand=time_varying,
        time_varying_probability=time_varying,
        inference_engine=engine,
    )
    forecaster.fit(y)

    fh = np.arange(1, 5)
    y_pred = forecaster.predict(fh=fh)

    assert y_pred.shape == fh.shape
