"""Poisson distribution cost function.

This module contains the PoissonCost class, which is a cost function for
change point detection based on the Poisson distribution. The cost is
twice the negative log likelihood of the Poisson distribution.
"""

import numpy as np

from ..utils.numba import njit
from ._utils import MeanType, check_non_negative_parameter
from .base import BaseCost


@njit
def poisson_log_likelihood(rate: float, samples: np.ndarray[np.integer]) -> float:
    """Fast Poisson log-likelihood for fixed rate parameter."""
    # Count the number of each sample value [0, 1, 2, ..., max_sample]:
    bin_counts = np.bincount(samples)
    # Reverse the counts and calculate the cumulative sum:
    sample_counts = np.flip(np.cumsum(np.flip(bin_counts)))
    max_sample = len(sample_counts) - 1

    if max_sample < 2:
        # Only observed '0'- or '1'-valued samples.
        sum_log_factorial_samples = 0
    else:
        sum_log_factorial_samples = np.sum(
            np.log(np.arange(2, max_sample + 1)) * sample_counts[2:]
        )

    return (
        -len(samples) * rate
        + np.log(rate) * np.sum(samples)
        - sum_log_factorial_samples
    )


@njit
def poisson_mle_rate_log_likelihood(mle_rate, samples: np.ndarray[np.integer]):
    """Fast Poisson log-likelihood for MLE rate parameter."""
    # Assume sample: np.ndarray[int]
    bin_counts = np.bincount(samples)
    sample_counts = np.flip(np.cumsum(np.flip(bin_counts)))
    max_sample = len(sample_counts) - 1

    if max_sample < 2:
        sum_log_factorial_samples = 0
    else:
        sum_log_factorial_samples = np.sum(
            np.log(np.arange(2, max_sample + 1)) * sample_counts[2:]
        )

    return (
        len(samples) * mle_rate * (np.log(mle_rate) - 1.0) - sum_log_factorial_samples
    )


@njit
def poisson_cost_mle_params(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Evaluate the Poisson cost with MLE parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array of integer counts.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        for col in range(n_columns):
            samples = X[start:end, col]
            # MLE for Poisson rate is the sample mean
            mle_rate = np.mean(samples)
            if mle_rate > 0:
                costs[i, col] = -2.0 * poisson_mle_rate_log_likelihood(
                    mle_rate, samples
                )
            else:
                # Handle the case where all samples are zero
                costs[i, col] = (
                    0.0  # Log-likelihood is 0 when all samples and rate are 0
                )

    return costs


@njit
def poisson_cost_fixed_params(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray, rates: np.ndarray
) -> np.ndarray:
    """Evaluate the Poisson cost for fixed rate parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array of integer counts.
    rates : np.ndarray
        Rate parameters of the Poisson distribution, one per column.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        for col in range(n_columns):
            samples = X[start:end, col]
            costs[i, col] = -2.0 * poisson_log_likelihood(rates[col], samples)

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
        "distribution_type": "Poisson",
        "supports_fixed_param": True,
    }

    def __init__(
        self,
        param: MeanType | None = None,
    ):
        super().__init__(param)
        self._rates = None

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method precomputes quantities that speed up the cost evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate cost on. Must be a 2D array of integer counts.
        y: None
            Ignored. Included for API consistency by convention.
        """
        # Check that X contains only integers:
        if not (np.issubdtype(X.dtype, np.integer) or np.issubdtype(X.dtype, int)):
            raise ValueError("PoissonCost requires integer data.")

        # Check that X contains only non-negative integers
        if np.any(X < 0):
            raise ValueError("PoissonCost requires non-negative integer data.")

        self._param = self._check_param(self.param, X)
        if self.param is not None:
            self._rates = self._param

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
            columns is equal to the number of columns in the input data.
        """
        return poisson_cost_mle_params(starts, ends, self._X)

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
            columns is equal to the number of columns in the input data.
        """
        return poisson_cost_fixed_params(starts, ends, self._X, self._rates)

    def _check_fixed_param(self, param, X: np.ndarray) -> np.ndarray:
        """Check if the fixed parameter is valid relative to the data.

        Parameters
        ----------
        param : float or array-like
            Fixed rate parameters of the Poisson distribution.
        X : np.ndarray
            Input data.

        Returns
        -------
        param: np.ndarray
            Fixed rate parameters for the cost calculation, one per column.
        """
        rates = check_non_negative_parameter(param, X)
        return rates

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        # Need at least 1 sample to estimate the rate parameter
        return 1

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
        # One rate parameter per column
        return p

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class
        """
        params = [
            {"param": None},
            {"param": 1.0},
        ]
        return params
