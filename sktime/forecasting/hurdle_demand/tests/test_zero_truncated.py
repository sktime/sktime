import pytest

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("numpyro", "jax", severity="none"):
    import jax.random as jrnd
    from numpyro.distributions import NegativeBinomial2, Poisson

from sktime.forecasting.hurdle_demand import HurdleDemandForecaster
from sktime.forecasting.hurdle_demand._truncated_discrete import TruncatedDiscrete
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(HurdleDemandForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "distribution",
    [
        ("poisson", {"rate": 1.0}),
        ("negative-binomial", {"mean": 1.0, "concentration": 1.0}),
    ],
)
def test_zero_truncated_distribution(distribution):
    dist_name, params = distribution

    if dist_name == "poisson":
        distribution = Poisson(**params)
    elif dist_name == "negative-binomial":
        distribution = NegativeBinomial2(**params)

    new_dist = TruncatedDiscrete(distribution)
    samples = new_dist.sample(jrnd.key(123), (1_000,))

    bad_log_prob = new_dist.log_prob(0)
    assert bad_log_prob == -float("inf"), "Log probability at zero should be -inf"

    assert (samples.min() > 0).all()
    assert (new_dist.log_prob(samples) > distribution.log_prob(samples)).all()

    return
