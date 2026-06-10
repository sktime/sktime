# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk, Tveten
"""Multivariate Gaussian likelihood cost."""

__author__ = ["johannvk", "Tveten"]

import numpy as np

from sktime.detection._costs._base import BaseCost
from sktime.detection._utils import (
    CovType,
    MeanType,
    check_cov,
    check_mean,
    log_det_covariance,
)


def _gaussian_ll_at_mle_params(X, start, end):
    """Gaussian log-likelihood at MLE parameters for a segment."""
    n = end - start
    p = X.shape[1]
    X_segment = X[start:end]
    log_det = log_det_covariance(X_segment)

    if np.isnan(log_det):
        raise RuntimeError(
            f"Covariance matrix of X[{start}:{end}] is not positive definite."
        )

    twice_ll = -n * p * np.log(2 * np.pi) - n * log_det - p * n
    return twice_ll / 2.0


def _gaussian_cost_mle_params(starts, ends, X):
    """Gaussian twice negative log-likelihood cost at MLE params."""
    num_starts = len(starts)
    costs = np.zeros((num_starts, 1))
    for i in range(num_starts):
        ll = _gaussian_ll_at_mle_params(X, starts[i], ends[i])
        costs[i, 0] = -2.0 * ll
    return costs


def _gaussian_ll_at_fixed_params(X, start, end, mean, log_det_cov, inv_cov):
    """Gaussian log-likelihood at fixed parameters for a segment."""
    n = end - start
    p = X.shape[1]
    X_centered = X[start:end] - mean
    quadratic_form = np.sum(X_centered @ inv_cov * X_centered, axis=1)
    twice_ll = -n * p * np.log(2 * np.pi) - n * log_det_cov - np.sum(quadratic_form)
    return twice_ll / 2.0


def _gaussian_cost_fixed_params(starts, ends, X, mean, log_det_cov, inv_cov):
    """Gaussian twice negative log-likelihood cost at fixed params."""
    num_starts = len(starts)
    costs = np.zeros((num_starts, 1))
    for i in range(num_starts):
        ll = _gaussian_ll_at_fixed_params(
            X, starts[i], ends[i], mean, log_det_cov, inv_cov
        )
        costs[i, 0] = -2.0 * ll
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

    def _evaluate_optim_param(self, X, starts, ends):
        return _gaussian_cost_mle_params(starts, ends, X)

    def _evaluate_fixed_param(self, X, starts, ends, param):
        mean, cov = param
        inv_cov = np.linalg.inv(cov)
        _, log_det_cov = np.linalg.slogdet(cov)
        return _gaussian_cost_fixed_params(starts, ends, X, mean, log_det_cov, inv_cov)

    def _check_fixed_param(self, param, X):
        if not isinstance(param, tuple) or len(param) != 2:
            raise ValueError("Fixed parameters must be (mean, covariance).")
        mean, cov = param
        mean = check_mean(mean, X)
        cov = check_cov(cov, X)
        return mean, cov

    @property
    def min_size(self):
        return None

    def get_model_size(self, p):
        return p + p * (p + 1) // 2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"param": None},
            {"param": (0.0, 1.0)},
            {"param": (np.zeros(1), np.eye(1))},
        ]
