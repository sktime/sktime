import jax
import jax.numpy as jnp
from numpyro.distributions import Bernoulli, Distribution, constraints
from numpyro.distributions.util import validate_sample

from ._truncated_discrete import TruncatedDiscrete


class HurdleDistribution(Distribution):
    """A Hurdle distribution.

    This distribution models data with an excess of zeros. It is a mixture of a
    point mass at zero and a positive distribution for values greater than zero.
    This implementation creates the positive distribution by internally truncating
    a given base distribution at `low=0`.

    Parameters
    ----------
    prob_gt_zero: jnp.ndarray
        Probability of observing a value greater than zero.

    positive_dist: TruncatedDiscrete
        Truncated distribution to use for positive values.
    """

    arg_constraints = {"prob_gt_zero": constraints.unit_interval}
    support = constraints.nonnegative
    has_rsample = False

    pytree_data_fields = ("prob_gt_zero", "positive_dist")

    def __init__(
        self,
        prob_gt_zero: jnp.ndarray,
        positive_dist: TruncatedDiscrete,
        validate_args=None,
    ):
        self.prob_gt_zero = prob_gt_zero
        self.positive_dist = positive_dist

        batch_shape = jnp.broadcast_shapes(
            jnp.shape(prob_gt_zero), self.positive_dist.batch_shape
        )

        super().__init__(
            batch_shape=batch_shape,
            event_shape=self.positive_dist.event_shape,
            validate_args=validate_args,
        )

    @validate_sample
    def log_prob(self, value):  # noqa: D102
        log_prob_zero = jnp.log1p(-self.prob_gt_zero)
        log_prob_positive = jnp.log(self.prob_gt_zero) + self.positive_dist.log_prob(
            value
        )

        is_zero = value == 0
        return jnp.where(is_zero, log_prob_zero, log_prob_positive)

    def sample(self, key, sample_shape=()):
        key_hurdle, key_positive = jax.random.split(key)
        is_positive = Bernoulli(probs=self.prob_gt_zero).sample(
            key_hurdle, sample_shape
        )

        positive_values = self.positive_dist.sample(key_positive, sample_shape)

        return jnp.where(is_positive, positive_values, 0)
