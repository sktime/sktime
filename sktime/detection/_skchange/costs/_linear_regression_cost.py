"""Linear Regression cost function.

This module contains the LinearRegressionCost class, which is a cost function for
change point detection based on linear regression. The cost is the sum of squared
residuals from fitting a linear regression model within each segment.
"""

import numpy as np

from ..utils.numba import njit
from ..utils.validation.parameters import check_data_column
from .base import BaseCost


@njit
def linear_regression_cost(X: np.ndarray, y: np.ndarray) -> float:
    """Compute the cost for a linear regression model.

    Parameters
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Target values.

    Returns
    -------
    cost : float
        Sum of squared residuals from the linear regression.
    """
    # Returns: (coeffs, residuals, X_rank, X_singular_values)
    coeffs, residuals, X_rank, _ = np.linalg.lstsq(X, y)

    # If rank(X) < X.shape[1], or X.shape[0] <= X.shape[1],
    # "residuals" is an empty array.
    if X_rank < X.shape[1]:
        # Underdetermined system, need to compute residuals manually.
        manual_residuals = y - X @ coeffs
        return np.sum(np.square(manual_residuals))
    else:
        # Full rank or overdetermined system, use lstsq residuals.
        return np.sum(residuals)


@njit
def linear_regression_cost_fixed_params(
    starts: np.ndarray,
    ends: np.ndarray,
    response_data: np.ndarray,
    covariate_data: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Evaluate the linear regression cost for fixed parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    response_data : np.ndarray
        Array containing the response variable values.
    covariate_data : np.ndarray
        Array containing the covariate/predictor values.
    coeffs : np.ndarray
        Fixed regression coefficients. Shape should be compatible with
        covariate_data for matrix multiplication.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs with one row for each interval and one column.
    """
    n_intervals = len(starts)
    costs = np.zeros((n_intervals, 1))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        X_features = covariate_data[start:end]
        y_response = response_data[start:end, None]

        # Compute predictions using fixed parameters:
        y_pred = X_features @ coeffs

        # Calculate residual sum of squares:
        costs[i, 0] = np.sum(np.square(y_response - y_pred))

    return costs


@njit
def linear_regression_cost_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    response_data: np.ndarray,
    covariate_data: np.ndarray,
) -> np.ndarray:
    """Evaluate the linear regression cost for each interval.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    response_data : np.ndarray
        Array containing the response variable values.
    covariate_data : np.ndarray
        Array containing the covariate/predictor values.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs with one row for each interval and one column.
    """
    n_intervals = len(starts)
    costs = np.zeros((n_intervals, 1))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        X_features = covariate_data[start:end]
        y_response = response_data[start:end]
        costs[i, 0] = linear_regression_cost(X_features, y_response)

    return costs


class LinearRegressionCost(BaseCost):
    """Linear Regression sum of squared residuals cost.

    This cost computes the sum of squared residuals from fitting a linear
    regression model within each segment. One column of the input data X is
    used as the response variable, and the remaining columns are used as
    predictors.

    Parameters
    ----------
    response_col : int, optional (default=0)
        Index of column in X to use as the response variable.
    covariate_cols : list of `int` or `str`, optional (default=None)
        Indices of columns in X to use as predictors. If None, all columns
        except the response column are used as predictors.
    param : array-like, optional (default=None)
        Fixed regression coefficients. If None, coefficients are estimated
        for each interval using ordinary least squares. Must be an array
        with the same length as `covariate_cols`, if provided, or the number of
        columns in X minus one if `covariate_cols` is None.
    """

    _tags = {
        "authors": ["johannvk"],
        "maintainers": "johannvk",
        "supports_fixed_param": True,
        "is_conditional": True,  # The cost uses covariates.
        "is_aggregated": True,  # Only a single response column is supported.
    }

    def __init__(
        self,
        response_col: str | int,
        covariate_cols: list[str | int] = None,
        param=None,
    ):
        super().__init__(param)

        self.response_col = response_col
        self.covariate_cols = covariate_cols

        self._response_col_idx: int | None = None
        self._covariate_col_indices: list[int] | None = None

        self._response_data: np.ndarray | None = None
        self._covariate_data: np.ndarray | None = None

        self._coeffs: np.ndarray | None = None

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method validates input data and stores it for cost evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        # Check that X has enough columns
        if X.shape[1] <= 1:
            raise ValueError(
                "X must have at least 2 columns for linear regression "
                "(1 for response and at least 1 for predictors)."
            )

        # Check that response_col is valid:
        self._response_col_idx = check_data_column(
            self.response_col, "Response", X, self._X_columns
        )
        if self.covariate_cols is not None:
            self._covariate_col_indices = [
                check_data_column(col, "Covariate", X, self._X_columns)
                for col in self.covariate_cols
            ]
        else:
            # Use all columns except the response column as covariates:
            self._covariate_col_indices = list(range(X.shape[1]))
            self._covariate_col_indices.remove(self._response_col_idx)

        self._response_data = X[:, self._response_col_idx]
        self._covariate_data = X[:, self._covariate_col_indices]

        # Check params after input column indices are set:
        self._param = self._check_param(self.param, X)
        if self.param is not None:
            self._coeffs = self._param

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the linear regression cost for each interval.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.
        The cost is computed using the optimal L2 regression coefficients.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs with one row for each interval and one column.
        """
        return linear_regression_cost_intervals(
            starts, ends, self._response_data, self._covariate_data
        )

    def _evaluate_fixed_param(self, starts, ends) -> np.ndarray:
        """Evaluate the cost for fixed regression coefficients.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.
        The cost is computed using the fixed regression coefficients provided.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs with one row for each interval and one column.
        """
        return linear_regression_cost_fixed_params(
            starts, ends, self._response_data, self._covariate_data, self._coeffs
        )

    def _check_fixed_param(self, param, X: np.ndarray) -> np.ndarray:
        """Check if the fixed parameter is valid relative to the data.

        Parameters
        ----------
        param : array-like
            Fixed regression coefficients.
        X : np.ndarray
            Input data.

        Returns
        -------
        param: np.ndarray
            Fixed regression coefficients for cost calculation.
        """
        param = np.asarray(param)
        expected_length = len(self._covariate_col_indices)

        if param.size != expected_length:
            raise ValueError(
                f"Expected {expected_length} coefficients"
                f" ({expected_length} predictors), got {param.size}."
            )

        if param.ndim != 1 and param.shape[1] != 1:
            raise ValueError(
                f"Coefficients must have shape ({expected_length}, 1)  or"
                f" ({expected_length},). Got shape {param.shape}."
            )

        return param.reshape(-1, 1)

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        if self.is_fitted:
            # Need at least as many samples as covariates:
            return len(self._covariate_col_indices)
        else:
            return None

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
        # Number of parameters = all features except response variable.
        if self.is_fitted:
            # Could use fewer covariates than the total number of columns:
            return len(self._covariate_col_indices)
        else:
            # Default to all columns except the response variable:
            return p - 1

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
            {"param": None, "response_col": 0},
            {"param": np.array([1.0, 0.5, -0.3]), "response_col": 1},
        ]
        return params
