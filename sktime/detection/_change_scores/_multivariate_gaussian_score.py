# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Multivariate Gaussian change score."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._costs._multivariate_gaussian_cost import MultivariateGaussianCost
from sktime.detection.base._base_interval_scorer import BaseIntervalScorer


def _half_integer_digamma(twice_n):
    """Digamma function for half-integer arguments ``twice_n / 2``."""
    if twice_n % 2 == 0:
        res = -np.euler_gamma
        n = twice_n // 2
        for k in range(n - 1):
            res += 1.0 / (k + 1.0)
    else:
        res = -2 * np.log(2) - np.euler_gamma
        n = (twice_n - 1) // 2
        for k in range(1, n + 1):
            res += 2.0 / (2.0 * k - 1.0)
    return res


def _likelihood_ratio_expected_value(n, k, p):
    """Compute expected value of twice the negative log-likelihood ratio."""
    g = p * (
        np.log(2)
        + (n - 1) * np.log(n - 1)
        - (n - k - 1) * np.log(n - k - 1)
        - (k - 1) * np.log(k - 1)
    )
    for j in range(1, p + 1):
        g += (
            (n - 1) * _half_integer_digamma(n - j)
            - (k - 1) * _half_integer_digamma(k - j)
            - (n - k - 1) * _half_integer_digamma(n - k - j)
        )
    return g


def _bartlett_corrections(seq_lengths, cut_points, dimension):
    """Compute Bartlett correction factors."""
    corrections = np.zeros((len(seq_lengths), 1))
    for i in range(len(seq_lengths)):
        g = _likelihood_ratio_expected_value(seq_lengths[i], cut_points[i], dimension)
        corrections[i, 0] = dimension * (dimension + 3.0) / g
    return corrections


class MultivariateGaussianScore(BaseIntervalScorer):
    """Multivariate Gaussian change score for a change in mean and/or covariance.

    Scores are likelihood ratio scores for a change in mean and
    covariance under a multivariate Gaussian distribution [1]_.

    Parameters
    ----------
    apply_bartlett_correction : bool, default=True
        Whether to apply the Bartlett correction.

    References
    ----------
    .. [1] Zamba, K. D., & Hawkins, D. M. (2009). A Multivariate Change-Point
       Model for Change in Mean Vector and/or Covariance Structure.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "task": "change_score",
        "is_aggregated": True,
    }

    def __init__(self, apply_bartlett_correction=True):
        self.apply_bartlett_correction = apply_bartlett_correction
        super().__init__()
        self._cost = MultivariateGaussianCost()

    @property
    def min_size(self):
        return None

    def get_model_size(self, p):
        return p + p * (p + 1) // 2

    def _evaluate(self, X, cuts):
        total = cuts[:, [0, 2]]
        left = cuts[:, [0, 1]]
        right = cuts[:, [1, 2]]
        raw_scores = self._cost.evaluate(X, total) - (
            self._cost.evaluate(X, left) + self._cost.evaluate(X, right)
        )

        if self.apply_bartlett_correction:
            seg_lengths = cuts[:, 2] - cuts[:, 0]
            seg_splits = cuts[:, 1] - cuts[:, 0]
            corrections = _bartlett_corrections(seg_lengths, seg_splits, X.shape[1])
            return corrections * raw_scores
        return raw_scores

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"apply_bartlett_correction": False},
            {"apply_bartlett_correction": True},
        ]
