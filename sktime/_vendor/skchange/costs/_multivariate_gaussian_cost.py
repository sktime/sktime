"""Multivariate Gaussian likelihood cost."""

__author__ = ["johannvk", "Tveten"]
__all__ = ["MultivariateGaussianCost"]

import numpy as np

from ..utils.numba import njit, prange
from ..utils.numba.stats import log_det_covariance
from ._utils import CovType, MeanType, check_cov, check_mean
from .base import BaseCost


@njit
def _gaussian_ll_at_mle_params(
    X: np.ndarray,
    start: int,
    end: int,
) -> float:
    """Calculate the Gaussian log likelihood at the MLE parameters for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment (inclusive).
    end : int
        End index of the segment (exclusive).

    Returns
    -------
    log_likelihood : float
        The log likelihood of the observations in the
        interval ``[start, end)`` in the data matrix `X`,
        evaluated at the maximum likelihood parameter
        estimates for the mean and covariance matrix.
    """
    n = end - start
    p = X.shape[1]

    X_segment = X[start:end]
    log_det_cov = log_det_covariance(X_segment)

    if np.isnan(log_det_cov):
        raise RuntimeError(
            f"The covariance matrix of X[{start}:{end}] is not positive definite."
            + " Quick and dirty fix: Add a tiny amount of random noise to the data."
        )

    twice_log_likelihood = -n * p * np.log(2 * np.pi) - n * log_det_cov - p * n
    return twice_log_likelihood / 2.0


@njit
def gaussian_cost_mle_params(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Calculate the Gaussian log likelihood at fixed parameters for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment (inclusive).
    end : int
        End index of the segment (exclusive).

    Returns
    -------
    costs : np.ndarray
        A 2D array of twice negated log likelihood costs. One row for each interval,
        and a single column.
    """
    num_starts = len(starts)
    costs = np.zeros(num_starts).reshape(-1, 1)
    for i in prange(num_starts):
        segment_log_likelihood = _gaussian_ll_at_mle_params(X, starts[i], ends[i])
        costs[i, 0] = -2.0 * segment_log_likelihood
    return costs


@njit
def _gaussian_ll_at_fixed_params(
    X: np.ndarray,
    start: int,
    end: int,
    mean: np.ndarray,
    log_det_cov: float,
    inv_cov: np.ndarray,
) -> float:
    """Calculate the Gaussian log likelihood at fixed parameters for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment (inclusive).
    end : int
        End index of the segment (exclusive).
    mean : np.ndarray
        Fixed mean for the cost calculation.
    log_det_cov : float
        Log determinant of the fixed covariance matrix.
    inv_cov : np.ndarray
        Inverse of the fixed covariance matrix.

    Returns
    -------
    log_likelihood : float
        The log likelihood of the observations in the
        interval ``[start, end)`` in the data matrix `X`,
        evaluated at the fixed mean and covariance matrix
        parameters provided.
    """
    n = end - start
    p = X.shape[1]

    X_segment = X[start:end]
    X_centered = X_segment - mean
    quadratic_form = np.sum(X_centered @ inv_cov * X_centered, axis=1)
    twice_log_likelihood = (
        -n * p * np.log(2 * np.pi) - n * log_det_cov - np.sum(quadratic_form)
    )
    return twice_log_likelihood / 2.0


@njit
def gaussian_cost_fixed_params(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    mean: np.ndarray,
    log_det_cov: float,
    inv_cov: np.ndarray,
) -> np.ndarray:
    """Calculate the Gaussian log likelihood at fixed parameters for a segment.

    Parameters
    ----------
    mean : np.ndarray
        Fixed mean for the cost calculation.
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    log_det_cov : float
        Log determinant of the fixed covariance matrix.
    inv_cov : np.ndarray
        Inverse of the fixed covariance matrix.

    Returns
    -------
    costs : np.ndarray
        A 2D array of twice negated log likelihood costs. One row for each interval,
        and a single column.
    """
    num_starts = len(starts)
    costs = np.zeros(num_starts).reshape(-1, 1)
    for i in prange(num_starts):
        segment_log_likelihood = _gaussian_ll_at_fixed_params(
            X, starts[i], ends[i], mean, log_det_cov, inv_cov
        )
        costs[i, 0] = -2.0 * segment_log_likelihood
    return costs


class MultivariateGaussianCost(BaseCost):
    """Multivariate Gaussian likelihood cost.

    Parameters
    ----------
    param : 2-tuple of float or np.ndarray, or None (default=None)
        Fixed mean and covariance matrix for the cost calculation.
        If ``None``, the maximum likelihood estimates are used.
    """

    _tags = {
        "authors": ["johannvk", "Tveten"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
        "is_aggregated": True,
    }

    def __init__(self, param: tuple[MeanType, CovType] | None = None):
        super().__init__(param)

    def _check_fixed_param(
        self, param: tuple[MeanType, CovType], X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Check if the fixed mean parameter is valid.

        Parameters
        ----------
        param : 2-tuple of float or np.ndarray
            Fixed mean and covariance matrix for the cost calculation.
        X : np.ndarray
            Input data.

        Returns
        -------
        mean : np.ndarray
            Fixed mean for the cost calculation.
        """
        mean, cov = param
        mean = check_mean(mean, X)
        cov = check_cov(cov, X)
        return mean, cov

    @property
    def min_size(self) -> int | None:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.
        """
        if self.is_fitted:
            return self.n_variables + 1
        else:
            return None

    def get_model_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return p + p * (p + 1) // 2

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

        if self._param is not None:
            self._mean, cov = self._param
            self._inv_cov = np.linalg.inv(cov)
            _, self._log_det_cov = np.linalg.slogdet(cov)

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal parameters.

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
            is 1 since the MultivariateGaussianCost is inherently multivariate.
        """
        return gaussian_cost_mle_params(starts, ends, self._X)

    def _evaluate_fixed_param(self, starts, ends):
        """Evaluate the cost for the fixed parameters.

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
            is 1 since the MultivariateGaussianCost is inherently multivariate.
        """
        return gaussian_cost_fixed_params(
            starts, ends, self._X, self._mean, self._log_det_cov, self._inv_cov
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
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
            {"param": (np.zeros(1), np.eye(1))},
        ]
        return params
