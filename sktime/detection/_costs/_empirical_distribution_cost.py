# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Empirical distribution cost function."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._costs._base import BaseCost
from sktime.detection._utils import (
    check_larger_than,
    col_cumsum,
    compute_finite_difference_derivatives,
)

# ---------------------------------------------------------------------------
# Quantile-point construction
# ---------------------------------------------------------------------------


def _make_mle_edf_quantile_points(X, num_quantiles):
    """Compute quantile points for approximating the EDF integral."""
    n_samples = X.shape[0]
    integrated_edf_scaling = -np.log(2 * n_samples - 1)
    q_range = np.arange(1, num_quantiles + 1)
    integration_quantiles = 1.0 / (
        1 + np.exp(integrated_edf_scaling * ((2 * q_range - 1) / num_quantiles - 1))
    )
    quantile_points = np.quantile(X, integration_quantiles, axis=0)
    quantile_values = np.tile(integration_quantiles.reshape(-1, 1), (1, X.shape[1]))
    return quantile_points, quantile_values


def _make_cumulative_edf_cache(xs, quantile_points):
    """Create a cumulative EDF cache for a single column."""
    lte_mask = (xs[:, np.newaxis] < quantile_points[np.newaxis, :]).astype(np.float64)
    lte_mask += 0.5 * (xs[:, np.newaxis] == quantile_points[np.newaxis, :])
    return col_cumsum(lte_mask, init_zero=True)


def _make_fixed_cdf_quantile_weights(fixed_quantiles, quantile_points):
    """Compute quantile integration weights for fixed CDF cost."""
    reciprocal_weights = 1.0 / (fixed_quantiles * (1.0 - fixed_quantiles))
    derivs = compute_finite_difference_derivatives(quantile_points, fixed_quantiles)
    return reciprocal_weights * derivs


# ---------------------------------------------------------------------------
# Cost evaluation (numpy, no numba)
# ---------------------------------------------------------------------------


def _mle_edf_cost_cached(cumulative_edf, starts, ends):
    """Approximate MLE EDF cost from cumulative EDF quantiles."""
    n_samples, num_quantiles = cumulative_edf.shape
    edf_scale = -np.log(2 * n_samples - 1)

    segment_edfs = (cumulative_edf[ends, :] - cumulative_edf[starts, :]) / (
        ends - starts
    ).reshape(-1, 1).astype(float)

    segment_edfs = np.clip(segment_edfs, 1e-10, 1 - 1e-10)
    one_minus = 1.0 - segment_edfs

    ll = np.sum(
        segment_edfs * np.log(segment_edfs) + one_minus * np.log(one_minus),
        axis=1,
    )
    ll *= (-2.0 * edf_scale / num_quantiles) * (ends - starts).astype(float)
    return -2.0 * ll


def _fixed_cdf_cost_cached(
    cumulative_edf,
    starts,
    ends,
    log_fixed_q,
    log_one_minus_fixed_q,
    quantile_weights,
):
    """EDF cost evaluated against a fixed reference CDF."""
    segment_edfs = (cumulative_edf[ends, :] - cumulative_edf[starts, :]) / (
        ends - starts
    ).reshape(-1, 1).astype(float)

    one_minus = 1.0 - segment_edfs

    ll = (ends - starts).astype(float) * np.sum(
        (segment_edfs * log_fixed_q + one_minus * log_one_minus_fixed_q)
        * quantile_weights[np.newaxis, :],
        axis=1,
    )
    return -2.0 * ll


class EmpiricalDistributionCost(BaseCost):
    """Empirical Distribution Cost.

    Approximate empirical distribution cost using the integrated log-likelihood
    of the empirical CDF [1]_.

    Parameters
    ----------
    param : tuple of (np.ndarray, np.ndarray), optional (default=None)
        If None, cost is evaluated on the MLE of the CDF.
        Otherwise, ``(fixed_samples, fixed_cdf_quantiles)`` to evaluate against.
    num_approximation_quantiles : int, optional (default=None)
        Number of quantiles used to approximate the EDF.
        If None, set to ``ceil(4 * log(n_samples))`` at evaluation time.

    References
    ----------
    .. [1] Haynes, K., Fearnhead, P. & Eckley, I.A. A computationally efficient
       nonparametric approach for changepoint detection. Stat Comput 27,
       1293-1305 (2017).
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "is_aggregated": False,
        "supports_fixed_param": True,
    }

    def __init__(self, param=None, num_approximation_quantiles=None):
        self.num_approximation_quantiles = num_approximation_quantiles
        check_larger_than(
            min_value=3,
            value=self.num_approximation_quantiles,
            name="num_approximation_quantiles",
            allow_none=True,
        )
        super().__init__(param)

    def _evaluate_optim_param(self, X, starts, ends):
        n_samples, n_cols = X.shape
        if self.num_approximation_quantiles is None:
            num_q = int(np.ceil(4 * np.log(n_samples)))
        else:
            num_q = self.num_approximation_quantiles

        q_points, _ = _make_mle_edf_quantile_points(X, num_q)

        costs = np.zeros((len(starts), n_cols))
        for col in range(n_cols):
            cache = _make_cumulative_edf_cache(X[:, col], q_points[:, col])
            costs[:, col] = _mle_edf_cost_cached(cache, starts, ends)
        return costs

    def _evaluate_fixed_param(self, X, starts, ends, param):
        q_points, q_values = param
        n_cols = X.shape[1]

        costs = np.zeros((len(starts), n_cols))
        for col in range(n_cols):
            cache = _make_cumulative_edf_cache(X[:, col], q_points[:, col])
            log_q = np.log(q_values[:, col])
            log_1mq = np.log(1.0 - q_values[:, col])
            weights = _make_fixed_cdf_quantile_weights(
                q_values[:, col], q_points[:, col]
            )
            costs[:, col] = _fixed_cdf_cost_cached(
                cache, starts, ends, log_q, log_1mq, weights
            )
        return costs

    def _check_fixed_param(self, param, X):
        fixed_samples, fixed_quantiles = param
        fixed_samples = np.asarray(fixed_samples, dtype=float)
        fixed_quantiles = np.asarray(fixed_quantiles, dtype=float)

        if fixed_samples.shape != fixed_quantiles.shape:
            raise ValueError("Samples and quantiles must have the same shape.")
        if fixed_samples.ndim == 1:
            fixed_samples = fixed_samples.reshape(-1, 1)
        if fixed_quantiles.ndim == 1:
            fixed_quantiles = fixed_quantiles.reshape(-1, 1)
        if not np.all(np.diff(fixed_samples, axis=0) > 0):
            raise ValueError("Fixed samples must be sorted and strictly increasing.")
        if not np.all(np.diff(fixed_quantiles, axis=0) > 0):
            raise ValueError("Fixed CDF quantiles must be sorted, strictly increasing.")
        if not (np.all(fixed_quantiles >= 0) and np.all(fixed_quantiles <= 1)):
            raise ValueError("Fixed quantiles must be in [0, 1].")

        fixed_quantiles = np.clip(fixed_quantiles, 1e-10, 1 - 1e-10)
        if fixed_samples.shape[1] == 1 and X.shape[1] > 1:
            fixed_samples = np.tile(fixed_samples, (1, X.shape[1]))
            fixed_quantiles = np.tile(fixed_quantiles, (1, X.shape[1]))

        return fixed_samples, fixed_quantiles

    @property
    def min_size(self):
        if self.num_approximation_quantiles is not None:
            return self.num_approximation_quantiles
        return 10

    def get_model_size(self, p):
        if self.num_approximation_quantiles is not None:
            return 2 * self.num_approximation_quantiles
        raise ValueError("Cost not evaluated yet; number of quantiles unknown.")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        fixed_samples = np.array(
            [
                -2.326,
                -1.96,
                -1.645,
                -1.036,
                -0.524,
                0.0,
                0.524,
                1.036,
                1.645,
                1.96,
                2.326,
            ]
        )
        fixed_quantiles = np.array(
            [0.01, 0.025, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.975, 0.99]
        )
        return [
            {"param": None, "num_approximation_quantiles": 10},
            {
                "param": (fixed_samples, fixed_quantiles),
                "num_approximation_quantiles": len(fixed_quantiles),
            },
        ]
