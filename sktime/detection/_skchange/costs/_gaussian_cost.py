"""Univariate Gaussian likelihood cost."""

__author__ = ["Tveten"]

import numpy as np

from ..utils.numba import njit
from ..utils.numba.general import truncate_below
from ..utils.numba.stats import col_cumsum
from ._utils import MeanType, VarType, check_mean, check_var
from .base import BaseCost


@njit
def var_from_sums(
    sums: np.ndarray, sums2: np.ndarray, starts: np.ndarray, ends: np.ndarray
):
    """Calculate variance from precomputed sums.

    Parameters
    ----------
    sums : np.ndarray
        Cumulative sums of ``X``, where the first row are ``0``s.
    sums2 : np.ndarray
        Cumulative sums of ``X**2``, where the first row are ``0``s.
    starts : np.ndarray
        Start indices in the original data ``X``.
    ends : np.ndarray
        End indices in the original data ``X``.

    Returns
    -------
    var : float
        Variance of ``X[i:j]``.
    """
    n = (ends - starts).reshape(-1, 1)
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    var = partial_sums2 / n - (partial_sums / n) ** 2
    return truncate_below(var, 1e-16)  # standard deviation lower bound of 1e-8


@njit
def gaussian_var_cost_optim(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for an optimal constant mean for each segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    sums : np.ndarray
        Cumulative sum of the input data, with a row of ``0``-entries as the first row.
    sums2 : np.ndarray
        Cumulative sum of the squared input data, with a row of ``0``-entries as the
        first row.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n = (ends - starts).reshape(-1, 1)
    var = var_from_sums(sums, sums2, starts, ends)
    log_likelihood = -n * np.log(2 * np.pi * var) - n
    return -log_likelihood


@njit
def gaussian_var_cost_fixed(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for a fixed constant mean for each segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    sums : np.ndarray
        Cumulative sum of the input data, with a row of ``0``-entries as the first row.
    sums2 : np.ndarray
        Cumulative sum of the squared input data, with a row of ``0``-entries as the
        first row.
    mean : np.ndarray
        Fixed mean for the cost calculation.
    var : np.ndarray
        Fixed variance for the cost calculation.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
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

    def __init__(self, param: tuple[MeanType, VarType] | None = None):
        super().__init__(param)

    def _check_fixed_param(
        self, param: tuple[MeanType, VarType], X: np.ndarray
    ) -> np.ndarray:
        """Check if the fixed mean parameter is valid.

        Parameters
        ----------
        param : 2-tuple of float or np.ndarray, or None (default=None)
            Fixed mean(s) and variance(s) for the cost calculation.
        X : np.ndarray
            Input data.

        Returns
        -------
        mean : np.ndarray
            Fixed mean for the cost calculation.
        """
        mean, var = param
        mean = check_mean(mean, X)
        var = check_var(var, X)
        return mean, var

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``intervals[i, 1] - intervals[i, 0]``.
        """
        return 2

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return 2 * p

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

        self._sums = col_cumsum(X, init_zero=True)
        self._sums2 = col_cumsum(X**2, init_zero=True)
        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal parameter.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of columns
            is equal to the number of columns in the input data, where each column
            represents the univariate cost for the corresponding input data column.
        """
        return gaussian_var_cost_optim(starts, ends, self._sums, self._sums2)

    def _evaluate_fixed_param(self, starts, ends):
        """Evaluate the cost for the fixed parameter.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of columns
            is equal to the number of columns in the input data, where each column
            represents the univariate cost for the corresponding input data column.
        """
        mean, var = self._param
        return gaussian_var_cost_fixed(starts, ends, self._sums, self._sums2, mean, var)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return "default" set.
            There are currently no reserved values for interval evaluators.

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
            {"param": (np.zeros(1), np.ones(1))},
        ]
        return params
