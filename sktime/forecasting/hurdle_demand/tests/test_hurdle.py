"""Tests for Hurdle estimator."""

import numpy as np
import pytest

from sktime.datasets import load_PBS_dataset
from sktime.forecasting.hurdle_demand import HurdleDemandForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(HurdleDemandForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("family", ["poisson", "negative-binomial"])
@pytest.mark.parametrize("time_varying", [True, False])
@pytest.mark.parametrize(
    "engine",
    [
        ("mcmc", {"num_samples": 5, "num_warmup": 50, "num_chains": 4}),
        ("map", {"num_steps": 10, "num_samples": 10}),
    ],
)
def test_hurdle_model(family: str, time_varying: bool, engine):
    from prophetverse.engine import MAPInferenceEngine, MCMCInferenceEngine
    from skpro.distributions import Hurdle, NegativeBinomial, Poisson

    """Test that Hurdle model can be instantiated and run with default parameters."""
    engine_type, kwargs = engine

    if engine_type == "mcmc":
        engine = MCMCInferenceEngine(**kwargs)
    elif engine_type == "map":
        engine = MAPInferenceEngine(**kwargs)

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

    y_dist = forecaster.predict_proba(fh=fh)

    assert isinstance(y_dist, Hurdle)

    if family == "poisson":
        assert isinstance(y_dist.distribution, Poisson)
    elif family == "negative-binomial":
        assert isinstance(y_dist.distribution, NegativeBinomial)
