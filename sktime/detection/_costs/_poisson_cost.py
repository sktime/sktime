# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Poisson distribution cost function."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._costs._base import BaseCost
from sktime.detection._utils import MeanType, check_non_negative_parameter


def _poisson_log_likelihood(rate, samples):
    """Fast Poisson log-likelihood for a fixed rate parameter."""
    bin_counts = np.bincount(samples)
    sample_counts = np.flip(np.cumsum(np.flip(bin_counts)))
    max_sample = len(sample_counts) - 1

    if max_sample < 2:
        sum_log_factorial_samples = 0.0
    else:
        sum_log_factorial_samples = np.sum(
            np.log(np.arange(2, max_sample + 1)) * sample_counts[2:]
        )

    return (
        -len(samples) * rate
        + np.log(rate) * np.sum(samples)
        - sum_log_factorial_samples
    )


def _poisson_mle_rate_log_likelihood(mle_rate, samples):
    """Fast Poisson log-likelihood for the MLE rate parameter."""
    bin_counts = np.bincount(samples)
    sample_counts = np.flip(np.cumsum(np.flip(bin_counts)))
    max_sample = len(sample_counts) - 1

    if max_sample < 2:
        sum_log_factorial_samples = 0.0
    else:
        sum_log_factorial_samples = np.sum(
            np.log(np.arange(2, max_sample + 1)) * sample_counts[2:]
        )

    return (
        len(samples) * mle_rate * (np.log(mle_rate) - 1.0) - sum_log_factorial_samples
    )


def _poisson_cost_mle_params(starts, ends, X):
    """Evaluate the Poisson cost with MLE parameters."""
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        for col in range(n_columns):
            samples = X[start:end, col].astype(np.int64)
            mle_rate = np.mean(samples)
            if mle_rate > 0:
                costs[i, col] = -2.0 * _poisson_mle_rate_log_likelihood(
                    mle_rate, samples
                )
            else:
                costs[i, col] = 0.0

    return costs


def _poisson_cost_fixed_params(starts, ends, X, rates):
    """Evaluate the Poisson cost for fixed rate parameters."""
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        for col in range(n_columns):
            samples = X[start:end, col].astype(np.int64)
            costs[i, col] = -2.0 * _poisson_log_likelihood(rates[col], samples)

    return costs


class PoissonCost(BaseCost):
    """Poisson distribution twice negative log likelihood cost.

    Parameters
    ----------
    param : float or array-like, optional (default=None)
        Fixed rate parameter of the Poisson distribution.
        If None, the cost is evaluated with rate set to
        the MLE estimate (sample mean) over each interval.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
    }

    def __init__(self, param: MeanType | None = None):
        super().__init__(param)

    def _evaluate_optim_param(self, X, starts, ends):
        self._validate_poisson_data(X)
        return _poisson_cost_mle_params(starts, ends, X)

    def _evaluate_fixed_param(self, X, starts, ends, param):
        self._validate_poisson_data(X)
        return _poisson_cost_fixed_params(starts, ends, X, param)

    def _validate_poisson_data(self, X):
        if not (np.issubdtype(X.dtype, np.integer) or np.all(X == X.astype(int))):
            raise ValueError("PoissonCost requires integer data.")
        if np.any(X < 0):
            raise ValueError("PoissonCost requires non-negative integer data.")

    def _check_fixed_param(self, param, X):
        return check_non_negative_parameter(param, X)

    @property
    def min_size(self):
        return 1

    def get_model_size(self, p):
        return p

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"param": None}, {"param": 1.0}]
