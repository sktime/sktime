# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Laplace distribution cost function."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._costs._base import BaseCost
from sktime.detection._utils import (
    MeanType,
    VarType,
    check_mean,
    check_non_negative_parameter,
)


def _laplace_log_likelihood(centered_X, scales):
    """Log likelihood of a Laplace distribution (column-wise)."""
    n_samples = len(centered_X)
    return -n_samples * np.log(2 * scales) - np.sum(np.abs(centered_X), axis=0) / scales


def _laplace_cost_mle_params(starts, ends, X):
    """Evaluate the Laplace cost with MLE parameters."""
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        segment = X[start:end]
        mle_locations = np.median(segment, axis=0)
        centered_X = segment - mle_locations[np.newaxis, :]
        mle_scales = np.mean(np.abs(centered_X), axis=0)
        costs[i, :] = -2.0 * _laplace_log_likelihood(centered_X, mle_scales)

    return costs


def _laplace_cost_fixed_params(starts, ends, X, locations, scales):
    """Evaluate the Laplace cost with fixed parameters."""
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        centered_X = X[start:end] - locations[np.newaxis, :]
        costs[i, :] = -2.0 * _laplace_log_likelihood(centered_X, scales)

    return costs


class LaplaceCost(BaseCost):
    """Laplace distribution twice negative log likelihood cost.

    Parameters
    ----------
    param : tuple[MeanType, VarType], optional (default=None)
        Fixed location and scale parameters of the Laplace distribution.
        If None, the cost is evaluated with location and scale set to
        the MLE estimates of the parameters over each interval.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
    }

    def __init__(self, param: tuple[MeanType, VarType] | None = None):
        super().__init__(param)

    def _evaluate_optim_param(self, X, starts, ends):
        return _laplace_cost_mle_params(starts, ends, X)

    def _evaluate_fixed_param(self, X, starts, ends, param):
        locations, scales = param
        return _laplace_cost_fixed_params(starts, ends, X, locations, scales)

    def _check_fixed_param(self, param, X):
        if not isinstance(param, tuple) or len(param) != 2:
            raise ValueError("Fixed Laplace parameters must be (location, scale).")
        means = check_mean(param[0], X)
        scales = check_non_negative_parameter(param[1], X)
        return means, scales

    @property
    def min_size(self):
        return 2

    def get_model_size(self, p):
        return 2 * p

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"param": None}, {"param": (0.0, 1.0)}]
