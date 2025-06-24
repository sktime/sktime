import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import validate_sample

from ._inverse_functions import _inverse_neg_binom, _inverse_poisson

REGISTRY = {
    dist.Poisson: _inverse_poisson,
    dist.NegativeBinomial2: _inverse_neg_binom,
}


class TruncatedDiscrete(Distribution):
    """A wrapper to create a zero-truncated version of a discrete distribution.

    Takes a base distribution (like Poisson or NegativeBinomial) and
    modifies its log_prob and sample methods to enforce a support of {1, 2, 3, ...}.

    Parameters
    ----------
    base_dist: numpyro.distributions.Distribution
        A discrete distribution with non-negative support.

    low: int, default=0
        The lower bound of the support. The distribution will only consider
        values greater than this bound.
    """

    has_rsample = False
    pytree_data_fields = ("base_dist",)
    pytree_aux_fields = ("_icdf", "low")

    def __init__(
        self, base_dist: dist.Distribution, low: int = 0, *, validate_args=None
    ):
        if base_dist.support is not constraints.nonnegative_integer:
            raise ValueError("ZeroTruncated only works with discrete distributions!")

        if base_dist.__class__ not in REGISTRY:
            raise ValueError(
                f"Base distribution '{base_dist.__class__.__name__}' not supported!"
            )

        super().__init__(
            batch_shape=base_dist.batch_shape,
            event_shape=base_dist.event_shape,
            validate_args=validate_args,
        )

        self.base_dist = base_dist
        self.low = low
        self._icdf = REGISTRY[base_dist.__class__]

    @property
    def support(self):  # noqa: D102
        return constraints.integer_greater_than(self.low)

    @validate_sample
    def log_prob(self, value):  # noqa: D102
        is_invalid = value <= self.low

        log_prob_base = self.base_dist.log_prob(value)

        log_prob_at_zero = self.base_dist.log_prob(self.low)
        log_normalizer = jnp.log1p(-jnp.exp(log_prob_at_zero))

        log_prob_truncated = log_prob_base - log_normalizer

        return jnp.where(is_invalid, -jnp.inf, log_prob_truncated)

    def sample(self, key, sample_shape=()):  # noqa: D102
        shape = sample_shape + self.batch_shape

        dtype = jnp.result_type(float)
        finfo = jnp.finfo(dtype)
        minval = finfo.tiny

        u = random.uniform(key, shape=shape, minval=minval)

        return self.icdf(u)

    def icdf(self, u):  # noqa: D102
        result_shape = jax.ShapeDtypeStruct(u.shape, jnp.result_type(float))

        low_cdf = self.base_dist.cdf(self.low)
        normalizer = 1.0 - low_cdf
        x = normalizer * u + low_cdf

        result = jax.pure_callback(
            self._icdf,
            result_shape,
            *(self.base_dist, x),
        )
        return result.astype(jnp.result_type(int))
