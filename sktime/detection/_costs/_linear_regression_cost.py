# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: johannvk
"""Linear regression cost function."""

__author__ = ["johannvk"]

import numpy as np

from sktime.detection._costs._base import BaseCost


def _linear_regression_cost(X_features, y_response):
    """Sum of squared residuals from OLS fit."""
    coeffs, residuals, rank, _ = np.linalg.lstsq(X_features, y_response, rcond=None)
    if rank < X_features.shape[1]:
        manual_residuals = y_response - X_features @ coeffs
        return np.sum(np.square(manual_residuals))
    else:
        return np.sum(residuals)


def _linear_regression_cost_intervals(starts, ends, response_data, covariate_data):
    """OLS cost for each interval."""
    n_intervals = len(starts)
    costs = np.zeros((n_intervals, 1))
    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        costs[i, 0] = _linear_regression_cost(
            covariate_data[start:end], response_data[start:end]
        )
    return costs


def _linear_regression_cost_fixed(starts, ends, response_data, covariate_data, coeffs):
    """OLS cost at fixed coefficients for each interval."""
    n_intervals = len(starts)
    costs = np.zeros((n_intervals, 1))
    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        y_pred = covariate_data[start:end] @ coeffs
        costs[i, 0] = np.sum(np.square(response_data[start:end, np.newaxis] - y_pred))
    return costs


class LinearRegressionCost(BaseCost):
    """Linear Regression sum of squared residuals cost.

    One column of the input data X is used as the response variable,
    and the remaining columns are used as predictors.

    Parameters
    ----------
    response_col : int
        Index of column in X to use as the response variable.
    covariate_cols : list of int, optional (default=None)
        Indices of columns in X to use as predictors. If None, all columns
        except the response column are used as predictors.
    param : array-like, optional (default=None)
        Fixed regression coefficients.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
        "is_aggregated": True,
    }

    def __init__(self, response_col: int = 0, covariate_cols=None, param=None):
        self.response_col = response_col
        self.covariate_cols = covariate_cols
        super().__init__(param)

    def _resolve_columns(self, X):
        """Resolve response and covariate column indices from X."""
        n_cols = X.shape[1]
        if n_cols <= 1:
            raise ValueError("X must have at least 2 columns for linear regression.")
        resp_idx = self.response_col
        if self.covariate_cols is not None:
            cov_indices = list(self.covariate_cols)
        else:
            cov_indices = [c for c in range(n_cols) if c != resp_idx]
        return resp_idx, cov_indices

    def _evaluate_optim_param(self, X, starts, ends):
        resp_idx, cov_indices = self._resolve_columns(X)
        response_data = X[:, resp_idx]
        covariate_data = X[:, cov_indices]
        return _linear_regression_cost_intervals(
            starts, ends, response_data, covariate_data
        )

    def _evaluate_fixed_param(self, X, starts, ends, param):
        resp_idx, cov_indices = self._resolve_columns(X)
        response_data = X[:, resp_idx]
        covariate_data = X[:, cov_indices]
        return _linear_regression_cost_fixed(
            starts, ends, response_data, covariate_data, param
        )

    def _check_fixed_param(self, param, X):
        param = np.asarray(param)
        _, cov_indices = self._resolve_columns(X)
        expected_length = len(cov_indices)
        if param.size != expected_length:
            raise ValueError(
                f"Expected {expected_length} coefficients, got {param.size}."
            )
        return param.reshape(-1, 1)

    @property
    def min_size(self):
        return None

    def get_model_size(self, p):
        return p - 1

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"param": None, "response_col": 0},
            {"param": np.array([1.0, 0.5, -0.3]), "response_col": 1},
        ]
