"""Laplace distribution cost function.

This module contains the LaplaceCost class, which is a cost function for
change point detection based on the Laplace distribution. The cost is
twice the negative log likelihood of the Laplace distribution.
"""

import numpy as np

from ..costs._utils import (
    MeanType,
    VarType,
    check_mean,
    check_non_negative_parameter,
)
from ..utils.numba import njit
from ..utils.numba.stats import col_median
from .base import BaseCost


@njit
def laplace_log_likelihood(centered_X: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Evaluate the log likelihood of a Laplace distribution.

    Parameters
    ----------
    centered_X : np.ndarray
        Column of data to evaluate the log likelihood for.
    scales : float
        Scale parameter of the Laplace distribution.

    Returns
    -------
    log_likelihood : np.ndarray
        Log likelihood of centered_X sampled i.i.d from a Laplace distribution
        with location = 0.0 and scale parameter as specified, columnwise.
    """
    n_samples = len(centered_X)
    return -n_samples * np.log(2 * scales) - np.sum(np.abs(centered_X), axis=0) / scales


@njit
def laplace_cost_mle_params(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Evaluate the Laplace cost with MLE parameters.

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
    mle_scales = np.zeros(n_columns)

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        mle_locations = col_median(X[start:end], output_array=mle_locations)
        centered_X = X[start:end, :] - mle_locations[None, :]

        # Compute the MLE scale for each column.
        # (Numba friendly, cannot supply 'axis' argument to np.mean)
        for col in range(n_columns):
            mle_scales[col] = np.mean(np.abs(centered_X[:, col]))

        costs[i, :] = -2.0 * laplace_log_likelihood(centered_X, mle_scales)

    return costs


@njit
def laplace_cost_fixed_params(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    locations: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Evaluate the Laplace cost for fixed parameters.

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
        centered_X = X[start:end, :] - locations[None, :]

        # Compute log-likelihood of samples in each column.
        costs[i, :] = -2.0 * laplace_log_likelihood(centered_X, scales)

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

    def __init__(
        self,
        param: tuple[MeanType, VarType] | None = None,
    ):
        super().__init__(param)
        self._locations = None
        self._scales = None

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
        self._param = self._check_param(self.param, X)
        if self.param is not None:
            self._locations, self._scales = self._param

        # MLE of the Laplace distribution location parameter is the median.
        # Tricky to precompute the median, so we'll do it on the fly.
        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal parameters.

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
        return laplace_cost_mle_params(starts, ends, self._X)

    def _evaluate_fixed_param(self, starts, ends) -> np.ndarray:
        """Evaluate the cost for the fixed parameters.

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
        return laplace_cost_fixed_params(
            starts, ends, self._X, self._locations, self._scales
        )

    def _check_fixed_param(
        self, param: tuple[MeanType, VarType], X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Check if the fixed parameter is valid relative to the data.

        Parameters
        ----------
        param : tuple[MeanType, VarType]
            Fixed location and scale parameters of the Laplace distribution.
        X : np.ndarray
            Input data.

        Returns
        -------
        param: tuple[np.ndarray, np.ndarray]
            Fixed location and scale parameters for the cost calculation.
        """
        if not isinstance(param, tuple) or len(param) != 2:
            raise ValueError("Fixed Laplace parameters must be (location, scale).")
        means = check_mean(param[0], X)
        scales = check_non_negative_parameter(param[1], X)
        return means, scales

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.

        Returns
        -------
        int or None
            The minimum valid size of an interval to evaluate. If ``None``, it is
            unknown what the minimum size is. E.g., the scorer may need to be fitted
            first to determine the minimum size.
        """
        # Need at least 2 samples to estimate the location and scale.
        return 2

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.

        Returns
        -------
        int
            Number of parameters in the cost function.
        """
        return 2 * p

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
            {"param": (0.0, 1.0)},
        ]
        return params
