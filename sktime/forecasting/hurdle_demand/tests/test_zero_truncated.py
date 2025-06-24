import jax.random as jrnd
import pytest
from numpyro.distributions import NegativeBinomial2, Poisson

from sktime.forecasting.hurdle_demand import HurdleDemandForecaster
from sktime.forecasting.hurdle_demand._truncated_discrete import TruncatedDiscrete
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(HurdleDemandForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("distribution", [Poisson(1.0), NegativeBinomial2(1.0, 1.0)])
@pytest.mark.parametrize("prng_key", [jrnd.key(123)])
def test_zero_truncated_distribution(distribution, prng_key):
    new_dist = TruncatedDiscrete(distribution)
    samples = new_dist.sample(prng_key, (1_000,))

    bad_log_prob = new_dist.log_prob(0)
    assert bad_log_prob == -float("inf"), "Log probability at zero should be -inf"

    assert (samples.min() > 0).all()
    assert (new_dist.log_prob(samples) > distribution.log_prob(samples)).all()

    return
