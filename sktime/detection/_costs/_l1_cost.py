# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""L1 (absolute difference) cost function."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._costs._base import BaseCost
from sktime.detection._utils import check_mean


def _l1_cost_mle_location(starts, ends, X):
    """Evaluate L1 cost for optimal location (median) per segment."""
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        segment = X[start:end]
        median = np.median(segment, axis=0)
        costs[i, :] = np.sum(np.abs(segment - median[None, :]), axis=0)

    return costs


def _l1_cost_fixed_location(starts, ends, X, locations):
    """Evaluate L1 cost for a fixed location per segment."""
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        centered_X = np.abs(X[start:end, :] - locations[None, :])
        costs[i, :] = np.sum(centered_X, axis=0)

    return costs


class L1Cost(BaseCost):
    """L1 cost function.

    Parameters
    ----------
    param : float or array-like, optional (default=None)
        Fixed mean for the cost calculation. If ``None``, the optimal mean is
        calculated as the median of each variable, for each interval.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
    }

    def __init__(self, param=None):
        super().__init__(param)

    def _check_fixed_param(self, param, X):
        return check_mean(param, X)

    def _evaluate_optim_param(self, X, starts, ends):
        """Evaluate cost with optimal (median) location."""
        return _l1_cost_mle_location(starts, ends, X)

    def _evaluate_fixed_param(self, X, starts, ends, param):
        """Evaluate cost with fixed location."""
        return _l1_cost_fixed_location(starts, ends, X, param)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params = [
            {"param": None},
            {"param": 0.0},
            {"param": np.array(1.0)},
        ]
        return params
