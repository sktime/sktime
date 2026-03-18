"""The CUSUM test statistic for a change in the mean."""

__author__ = ["Tveten"]

import numpy as np

from ..base import BaseIntervalScorer
from ..utils.numba import njit
from ..utils.numba.stats import col_cumsum


@njit
def cusum_score(
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
    sums: np.ndarray,
) -> np.ndarray:
    """
    Calculate the CUSUM score for a change in the mean.

    Compares the mean of the data before and after the split within the interval from
    ``start:end``.

    Parameters
    ----------
    starts : `np.ndarray`
        Start indices of the intervals to test for a change in the mean.
    ends : `np.ndarray`
        End indices of the intervals to test for a change in the mean.
    splits : `np.ndarray`
        Split indices of the intervals to test for a change in the mean.
    sums : `np.ndarray`
        Cumulative sum of the input data, with a row of 0-entries as the first row.

    Returns
    -------
    `np.ndarray`
        CUSUM scores for the intervals and splits.
    """
    n = ends - starts
    before_n = splits - starts
    after_n = ends - splits
    before_sum = sums[splits] - sums[starts]
    after_sum = sums[ends] - sums[splits]
    before_weight = np.sqrt(after_n / (n * before_n)).reshape(-1, 1)
    after_weight = np.sqrt(before_n / (n * after_n)).reshape(-1, 1)
    cusum = np.abs(before_weight * before_sum - after_weight * after_sum)
    return cusum


class CUSUM(BaseIntervalScorer):
    """CUSUM change score for a change in the mean.

    The classical CUSUM test statistic for a change in the mean is calculated as the
    weighted difference between the mean before and after a split point within an
    interval. See e.g. Section 4 of [2]_, the idea goes back to [1]_.

    References
    ----------
    .. [1] Page, E. S. (1954). Continuous inspection schemes. Biometrika, 41(1/2),
      100-115.

    .. [2] Wang, D., Yu, Y., & Rinaldo, A. (2020). Univariate mean change point
      detection: Penalization, cusum and optimality. Electronic Journal of Statistics,
      14(1) 1917-1961.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "task": "change_score",
    }

    def __init__(self):
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return 1

    def _fit(self, X: np.ndarray, y=None):
        """Fit the change score evaluator.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.
        """
        self._sums = col_cumsum(X, init_zero=True)
        return self

    def _evaluate(self, cuts: np.ndarray):
        """Evaluate the change score for a split within an interval.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with three columns of integer locations.
            The first column is the ``start``, the second is the ``split``, and the
            third is the ``end`` of the interval to evaluate.
            The difference between subsets ``X[start:split]`` and ``X[split:end]`` is
            evaluated for each row in `cuts`.

        Returns
        -------
        scores : np.ndarray
            A 2D array of change scores. One row for each cut. The number of
            columns is 1 if the change score is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the score is
            univariate. In this case, each column represents the univariate score for
            the corresponding input data column.
        """
        starts = cuts[:, 0]
        splits = cuts[:, 1]
        ends = cuts[:, 2]
        return cusum_score(starts, ends, splits, self._sums)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for interval scorers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # CUSUM does not have any parameters to set
        params = [{}]
        return params
