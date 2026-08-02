# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""CUSUM change score for a change in the mean."""

__author__ = ["Tveten"]

import numpy as np

from sktime.detection._utils import col_cumsum
from sktime.detection.base._base_interval_scorer import BaseIntervalScorer


def _cusum_score(starts, ends, splits, sums):
    """Calculate the CUSUM score for a change in the mean.

    Parameters
    ----------
    starts, ends, splits : np.ndarray
        Integer index arrays.
    sums : np.ndarray
        Cumulative sums with a row of zeros prepended.

    Returns
    -------
    np.ndarray
        CUSUM scores, shape ``(n_intervals, n_columns)``.
    """
    n = ends - starts
    before_n = splits - starts
    after_n = ends - splits
    before_sum = sums[splits] - sums[starts]
    after_sum = sums[ends] - sums[splits]
    before_weight = np.sqrt(after_n / (n * before_n)).reshape(-1, 1)
    after_weight = np.sqrt(before_n / (n * after_n)).reshape(-1, 1)
    return np.abs(before_weight * before_sum - after_weight * after_sum)


class CUSUM(BaseIntervalScorer):
    """CUSUM change score for a change in the mean.

    The classical CUSUM test statistic for a change in the mean [1]_ [2]_.

    References
    ----------
    .. [1] Page, E. S. (1954). Continuous inspection schemes. Biometrika.
    .. [2] Wang, D., Yu, Y., & Rinaldo, A. (2020). Univariate mean change
       point detection. Electronic Journal of Statistics.
    """

    _tags = {
        "authors": ["Tveten"],
        "maintainers": "Tveten",
        "task": "change_score",
    }

    def __init__(self):
        super().__init__()

    @property
    def min_size(self):
        return 1

    def _evaluate(self, X, cuts):
        sums = col_cumsum(X, init_zero=True)
        starts, splits, ends = cuts[:, 0], cuts[:, 1], cuts[:, 2]
        return _cusum_score(starts, ends, splits, sums)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{}]
