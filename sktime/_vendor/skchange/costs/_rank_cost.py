"""Difference in mean rank aggregated multivariate cost."""

import numpy as np
from scipy.linalg import pinvh

from ..utils.numba import njit
from .base import BaseCost


@njit
def _rank_cost(
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    centered_data_ranks: np.ndarray,
    pinv_rank_cov: np.ndarray,
) -> np.ndarray:
    """Compute the rank cost for segments.

    This function computes the rank cost for each segment defined by the
    start and end indices. The rank cost is based on the mean ranks of the
    data within each segment and the pseudo-inverse of the rank covariance
    matrix.

    Parameters
    ----------
    segment_starts : np.ndarray
        The start indices of the segments.
    segment_ends : np.ndarray
        The end indices of the segments.
    centered_data_ranks : np.ndarray
        The centered data ranks.
    pinv_rank_cov : np.ndarray
        The pseudo-inverse of the rank covariance matrix.

    Returns
    -------
    np.ndarray
        The rank costs for each segment.
    """
    n_samples, n_variables = centered_data_ranks.shape
    costs = np.zeros(segment_starts.shape[0])

    # Compute mean ranks for each segment:
    mean_segment_ranks = np.zeros(n_variables)
    normalization_constant = 4.0 / np.square(n_samples)

    for i, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends)):
        for var in range(n_variables):
            mean_segment_ranks[var] = np.mean(
                centered_data_ranks[segment_start:segment_end, var]
            )
        rank_score = (segment_end - segment_start) * (
            mean_segment_ranks.T @ pinv_rank_cov @ mean_segment_ranks
        )
        costs[i] = -rank_score * normalization_constant

    return costs


def _compute_ranks_and_pinv_cdf_cov(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the pseudo-inverse of the CDF covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Input data array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The pseudo-inverse of the rank covariance matrix and centered data ranks.
    """
    n_samples, n_variables = X.shape
    X_sorted_by_column = np.sort(X, axis=0)
    # Compute the Empirical CDF value for each column:
    data_ranks = np.zeros_like(X, dtype=np.float64)

    for col in range(n_variables):
        # Compute upper right ranks: (a[i-1] < v <= a[i])
        # One-indexed lower ranks:
        data_ranks[:, col] = 1 + np.searchsorted(
            X_sorted_by_column[:, col], X[:, col], side="left"
        )

        # Must average lower and upper ranks to work on data with duplicates:
        # Add lower left ranks: (a[i-1] <= v < a[i])
        data_ranks[:, col] += np.searchsorted(
            X_sorted_by_column[:, col], X[:, col], side="right"
        )
        data_ranks[:, col] /= 2

    cdf_values = np.copy(data_ranks) / n_samples
    centered_cdf_values = cdf_values - 0.5

    cdf_cov = 4.0 * (centered_cdf_values.T @ centered_cdf_values) / n_samples
    cdf_cov = cdf_cov.reshape(n_variables, n_variables)
    pinv_cdf_cov = pinvh(cdf_cov)

    centered_data_ranks = data_ranks - (n_samples + 1) / 2.0

    return centered_data_ranks, pinv_cdf_cov


class RankCost(BaseCost):
    """Rank based multivariate cost.

    This cost function uses mean rank statistics to detect changes in the
    distribution of multivariate data. Aggregates mean rank statistics over all
    variables using the pseudo-inverse of the covariance of the empirical CDF [1]_.

    Fulfills the PELT assumption that the summed cost of two adjacent segments is
    less than or equal to the cost of the combined segment, for K = 0.

    Parameters
    ----------
    param : any, optional (default=None)
        Not used. Included for API consistency by convention.

    References
    ----------
    .. [1] Lung-Yut-Fong, A., Lévy-Leduc, C., & Cappé, O. (2015). Homogeneity and
       change-point detection tests for multivariate data using rank statistics.
       Journal de la société française de statistique, 156(4), 133-162.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": ["johannvk"],
        "distribution_type": "None",
        "is_conditional": False,
        "is_aggregated": True,
        "supports_fixed_param": False,
    }

    def __init__(self, param=None):
        super().__init__(param)

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

        # Precompute rank statistics and covariance matrix
        self._centered_data_ranks, self._pinv_rank_cov = (
            _compute_ranks_and_pinv_cdf_cov(X)
        )

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal parameters.

        Evaluates the cost for `X[start:end]` for each start, end in starts, ends.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs with shape (n_intervals, 1) since this is an
            aggregated multivariate cost.
        """
        costs = _rank_cost(starts, ends, self._centered_data_ranks, self._pinv_rank_cov)
        return costs.reshape(-1, 1)

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
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
            Number of parameters in the cost function (covariance matrix parameters).
        """
        return p * (p + 1) // 2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params = [{}, {}]
        return params
