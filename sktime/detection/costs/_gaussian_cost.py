# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Univariate Gaussian likelihood cost."""

__author__ = ["Tveten"]

import numpy as np

from sktime.detection._costs._base import BaseCost
from sktime.detection._utils import check_mean, check_var, col_cumsum, truncate_below


def _var_from_sums(sums, sums2, starts, ends):
    """Calculate variance from precomputed sums."""
    n = (ends - starts).reshape(-1, 1)
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    var = partial_sums2 / n - (partial_sums / n) ** 2
    return truncate_below(var, 1e-16)


def _gaussian_var_cost_optim(starts, ends, sums, sums2):
    """Calculate Gaussian cost for optimal mean and variance."""
    n = (ends - starts).reshape(-1, 1)
    var = _var_from_sums(sums, sums2, starts, ends)
    log_likelihood = -n * np.log(2 * np.pi * var) - n
    return -log_likelihood


def _gaussian_var_cost_fixed(starts, ends, sums, sums2, mean, var):
    """Calculate Gaussian cost for fixed mean and variance."""
    n = (ends - starts).reshape(-1, 1)
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    quadratic_form = partial_sums2 - 2 * mean * partial_sums + n * mean**2
    log_likelihood = -n * np.log(2 * np.pi * var) - quadratic_form / var
    return -log_likelihood


class GaussianCost(BaseCost):
    """Univariate Gaussian likelihood cost.

    Parameters
    ----------
    param : 2-tuple of float or np.ndarray, or None (default=None)
        Fixed mean(s) and variance(s) for the cost calculation.
        If ``None``, the maximum likelihood estimates are used.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "supports_fixed_param": True,
    }

    def __init__(self, param=None):
        super().__init__(param)

    def _check_fixed_param(self, param, X):
        mean, var = param
        mean = check_mean(mean, X)
        var = check_var(var, X)
        return mean, var

    @property
    def min_size(self):
        """Minimum size of the interval to evaluate."""
        return 2

    def get_model_size(self, p):
        """Get the number of parameters in the cost function."""
        return 2 * p

    def _evaluate_optim_param(self, X, starts, ends):
        """Evaluate cost with optimal parameters."""
        sums = col_cumsum(X, init_zero=True)
        sums2 = col_cumsum(X**2, init_zero=True)
        return _gaussian_var_cost_optim(starts, ends, sums, sums2)

    def _evaluate_fixed_param(self, X, starts, ends, param):
        """Evaluate cost with fixed mean and variance."""
        mean, var = param
        sums = col_cumsum(X, init_zero=True)
        sums2 = col_cumsum(X**2, init_zero=True)
        return _gaussian_var_cost_fixed(starts, ends, sums, sums2, mean, var)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params = [
            {"param": None},
            {"param": (0.0, 1.0)},
            {"param": (np.zeros(1), np.ones(1))},
        ]
        return params
