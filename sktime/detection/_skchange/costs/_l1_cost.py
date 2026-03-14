"""L1 (absolute difference) cost function.

This module contains the L1Cost class, which is a cost function for
change point detection based on the L1 (absolute difference) cost.
"""

import numpy as np

from ..utils.numba import njit
from ..utils.numba.stats import col_median
from ._utils import MeanType, check_mean
from .base import BaseCost


@njit
def l1_cost_mle_location(starts: np.ndarray, ends: np.ndarray, X: np.ndarray):
    """Evaluate the L1 cost for a known scale.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))
    mle_locations = np.zeros(n_columns)

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        mle_locations = col_median(X[start:end], output_array=mle_locations)
        costs[i, :] = np.sum(np.abs(X[start:end] - mle_locations[None, :]), axis=0)

    return costs


@njit
def l1_cost_fixed_location(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray, locations: np.ndarray
):
    """Evaluate the L1 cost for a known scale.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
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

    def __init__(self, param: MeanType | None = None):
        super().__init__(param)
        self._mean = None

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method precomputes quantities that speed up the cost evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        self._mean = self._check_param(self.param, X)
        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal parameter.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is equal to the number of columns in the input data
            passed to `.fit()`. Each column represents the univariate
            cost for the corresponding input data column.
        """
        return l1_cost_mle_location(
            starts,
            ends,
            self._X,
        )

    def _evaluate_fixed_param(self, starts, ends):
        """Evaluate the cost for the fixed parameter.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is equal to the number of columns in the input data
            passed to `.fit()`. Each column represents the univariate
            cost for the corresponding input data column.
        """
        return l1_cost_fixed_location(starts, ends, self._X, self._mean)

    def _check_fixed_param(self, param: MeanType, X: np.ndarray) -> np.ndarray:
        """Check if the fixed parameter is valid relative to the data.

        Parameters
        ----------
        param : float
            Fixed parameter for the cost calculation.
        X : np.ndarray
            Input data.

        Returns
        -------
        param: np.ndarray
            Fixed parameter for the cost calculation.
        """
        return check_mean(param, X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for costs.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"param": None},
            {"param": 0.0},
            {"param": np.array(1.0)},
        ]
        return params
