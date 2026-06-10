# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Rank-based multivariate cost function."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._costs._base import BaseCost


def _rank_cost(segment_starts, segment_ends, centered_data_ranks, pinv_rank_cov):
    """Compute the rank cost for segments."""
    n_samples = centered_data_ranks.shape[0]
    costs = np.zeros(len(segment_starts))
    normalization_constant = 4.0 / np.square(n_samples)

    for i in range(len(segment_starts)):
        start, end = segment_starts[i], segment_ends[i]
        mean_ranks = np.mean(centered_data_ranks[start:end], axis=0)
        rank_score = (end - start) * (mean_ranks.T @ pinv_rank_cov @ mean_ranks)
        costs[i] = -rank_score * normalization_constant

    return costs


def _compute_ranks_and_pinv_cdf_cov(X):
    """Compute centered data ranks and pseudo-inverse of the CDF covariance matrix."""
    from scipy.linalg import pinvh

    n_samples, n_variables = X.shape
    X_sorted = np.sort(X, axis=0)
    data_ranks = np.zeros_like(X, dtype=np.float64)

    for col in range(n_variables):
        lower = 1 + np.searchsorted(X_sorted[:, col], X[:, col], side="left")
        upper = np.searchsorted(X_sorted[:, col], X[:, col], side="right")
        data_ranks[:, col] = (lower + upper) / 2.0

    cdf_values = data_ranks / n_samples
    centered_cdf = cdf_values - 0.5

    cdf_cov = 4.0 * (centered_cdf.T @ centered_cdf) / n_samples
    cdf_cov = cdf_cov.reshape(n_variables, n_variables)
    pinv_cdf_cov = pinvh(cdf_cov)

    centered_data_ranks = data_ranks - (n_samples + 1) / 2.0
    return centered_data_ranks, pinv_cdf_cov


class RankCost(BaseCost):
    """Rank based multivariate cost.

    Uses mean rank statistics to detect changes in the distribution
    of multivariate data, aggregating over all variables using the
    pseudo-inverse of the covariance of the empirical CDF [1]_.

    Parameters
    ----------
    param : any, optional (default=None)
        Not used. Included for API consistency by convention.

    References
    ----------
    .. [1] Lung-Yut-Fong, A., Levy-Leduc, C., & Cappe, O. (2015).
       Homogeneity and change-point detection tests for multivariate data
       using rank statistics.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": ["johannvk"],
        "is_aggregated": True,
        "supports_fixed_param": False,
    }

    def __init__(self, param=None):
        super().__init__(param)

    def _evaluate_optim_param(self, X, starts, ends):
        centered_data_ranks, pinv_rank_cov = _compute_ranks_and_pinv_cdf_cov(X)
        costs = _rank_cost(starts, ends, centered_data_ranks, pinv_rank_cov)
        return costs.reshape(-1, 1)

    @property
    def min_size(self):
        return 2

    def get_model_size(self, p):
        return p * (p + 1) // 2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{}, {}]
