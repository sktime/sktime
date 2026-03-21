"""Tests for LinearRegressionCost class."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from sktime.detection._skchange.change_detectors import PELT
from sktime.detection._skchange.costs import LinearRegressionCost


def test_linear_regression_cost_init():
    """Test initialization of LinearRegressionCost."""
    # Default parameters
    cost = LinearRegressionCost(response_col=0)
    assert cost.response_col == 0

    # Custom response_col
    cost = LinearRegressionCost(response_col=2)
    assert cost.response_col == 2

    # Invalid response_col type:
    with pytest.raises(ValueError):
        LinearRegressionCost(response_col="invalid").fit(np.random.rand(100, 3))


def test_linear_regression_cost_fit():
    """Test fitting of LinearRegressionCost."""
    # Create simple dataset
    X = np.random.rand(100, 3)

    # Valid fit
    cost = LinearRegressionCost(response_col=1)
    cost.fit(X)
    assert cost.is_fitted

    # Invalid number of columns
    X_single_col = np.random.rand(100, 1)
    with pytest.raises(ValueError):
        cost = LinearRegressionCost(response_col=0)
        cost.fit(X_single_col)

    # Invalid response_col
    with pytest.raises(ValueError):
        cost = LinearRegressionCost(response_col=5)  # Out of bounds
        cost.fit(X)


def test_linear_regression_cost_evaluate():
    """Test evaluation of LinearRegressionCost."""
    # Create regression dataset with known relationship
    X, y = make_regression(
        n_samples=100, n_features=3, n_informative=3, noise=0.1, random_state=42
    )
    # Add y as last column to X
    X_with_y = np.hstack((X, y.reshape(-1, 1)))

    # Fit the cost with y as the response (last column)
    cost = LinearRegressionCost(response_col=3)
    cost.fit(X_with_y)

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([100])
    cuts = np.hstack((starts, ends))
    costs = cost.evaluate(cuts=cuts)

    # Compare with sklearn's LinearRegression
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    expected_cost = np.sum((lr.predict(X) - y) ** 2)

    # Allow for small numerical differences:
    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_linear_regression_cost_evaluate_multiple_intervals():
    """Test evaluation of LinearRegressionCost on multiple intervals."""
    # Create data
    n_samples = 200
    X = np.random.rand(n_samples, 3)

    # Fit the cost:
    cost = LinearRegressionCost(response_col=0)
    cost.fit(X)

    # Define intervals
    starts = np.array([0, 50, 100, 150])
    ends = np.array([50, 100, 150, 200])
    cuts = np.hstack([starts.reshape(-1, 1), ends.reshape(-1, 1)])

    # Evaluate
    costs = cost.evaluate(cuts=cuts)

    # Check shape
    assert costs.shape == (4, 1)

    # Check all costs are non-negative
    assert np.all(costs >= 0)


def test_min_size_property():
    """Test the min_size property."""
    # Before fitting
    cost = LinearRegressionCost(response_col=0)
    assert cost.min_size is None

    # After fitting with 3 columns (response + 2 features)
    X = np.random.rand(100, 3)
    cost.fit(X)
    assert cost.min_size == 2  # 2 features


def test_get_model_size():
    """Test get_model_size method."""
    cost = LinearRegressionCost(response_col="log_house_price")
    # Number of parameters is equal to number of variables
    assert cost.get_model_size(5) == 4


def test_simple_linear_regression_cost_fixed_params():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float64)

    # y = 1 * x_0 + 2 * x_1 + 3:
    y: np.ndarray = np.dot(X, np.array([1, 2])) + 3.0

    reg = LinearRegression(fit_intercept=False).fit(X, y)

    fixed_coef = reg.coef_

    cost = LinearRegressionCost(param=fixed_coef, response_col=0)
    cost.fit(np.hstack((y.reshape(-1, 1), X)))

    # Test that number of parameters is equal to number of columns:
    assert cost.get_model_size(3) == 2

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([4])
    cuts = np.hstack((starts, ends))
    costs = cost.evaluate(cuts=cuts)

    # Calculate expected cost manually:
    y_pred = reg.predict(X)
    expected_cost = np.sum(np.square(y - y_pred))

    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_linear_regression_cost_fixed_params():
    """Test evaluation with fixed parameters."""
    # Create regression dataset
    X, y = make_regression(
        n_samples=10, n_features=2, n_informative=2, noise=0.1, random_state=42
    )
    X_with_y = np.hstack((X, y.reshape(-1, 1)))

    # First fit a regular linear regression to get coefficients
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)

    # Create coefficient array (intercept + coeffs)
    fixed_coeffs = lr.coef_

    # Create cost with fixed params
    cost = LinearRegressionCost(param=fixed_coeffs, response_col=2)
    cost.fit(X_with_y)

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([10])
    cuts = np.hstack((starts.reshape(-1, 1), ends.reshape(-1, 1)))
    costs = cost.evaluate(cuts=cuts)

    # Calculate expected cost manually
    y_pred = lr.predict(X)
    expected_cost = np.sum(np.square(y - y_pred))

    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_fixed_param_validation():
    """Test validation of fixed parameters."""
    # Create simple dataset
    X = np.random.rand(100, 3)

    # Valid parameters: (2 features excluding response)
    valid_params = np.array([0.2, -0.3])
    cost = LinearRegressionCost(param=valid_params, response_col=0)
    cost.fit(X)
    assert np.array_equal(cost._coeffs, valid_params.reshape(-1, 1))

    # Valid column vector params:
    valid_params_col = np.array([[0.2], [-0.3]])
    cost = LinearRegressionCost(param=valid_params_col, response_col=0)
    cost.fit(X)
    assert np.array_equal(cost._coeffs, valid_params_col)

    # Invalid parameter dimension (1, 2):
    invalid_params_2d = np.array([[0.2, 0.3]])
    with pytest.raises(ValueError):
        cost = LinearRegressionCost(param=invalid_params_2d, response_col=0)
        cost.fit(X)

    # Invalid parameter length
    invalid_params_length = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        cost = LinearRegressionCost(param=invalid_params_length, response_col=0)
        cost.fit(X)


def test_linear_regression_cost_with_pelt():
    """Test LinearRegressionCost on a structural change problem."""

    # Create a dataset with a structural change in the regression coefficients
    n_samples = 200
    np.random.seed(42)

    # Create predictors
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)

    # Create response with a change in coefficients at index 100
    y = np.zeros(n_samples)

    # First segment: y = 1*x1 + 0.5*x2 + noise
    y[:100] = 1.0 * x1[:100] + 0.5 * x2[:100] + np.random.normal(0, 0.5, 100)

    # Second segment: y = -1*x1 + 2*x2 + noise
    y[100:] = -1.0 * x1[100:] + 2.0 * x2[100:] + np.random.normal(0, 0.5, 100)

    # Create pandas DataFrame with named columns
    df = pd.DataFrame(
        {
            "response": y,
            "intercept": np.ones_like(y),
            "predictor1": x1,
            "predictor2": x2,
        }
    )

    # Create the cost function with the response column name
    cost = LinearRegressionCost(
        response_col="response",
        covariate_cols=["intercept", "predictor1", "predictor2"],
    )

    # Create and fit PELT change detector with the linear regression cost:
    pelt = PELT(cost=cost, min_segment_length=10)
    result = pelt.fit_predict(df)

    # Assert that the detected changepoint is close to the actual changepoint (100)
    assert len(result) == 1, "Expected exactly one changepoint"
    detected_cp = result.iloc[0].item()
    assert abs(detected_cp - 100) <= 1, (
        f"Detected changepoint {detected_cp} not close to actual (100)"
    )

    # Additional test: verify the coefficients differ between segments
    segment1 = df.iloc[:detected_cp]
    segment2 = df.iloc[detected_cp:]

    lr1 = LinearRegression(fit_intercept=False).fit(
        segment1[["intercept", "predictor1", "predictor2"]], segment1["response"]
    )
    lr2 = LinearRegression(fit_intercept=False).fit(
        segment2[["intercept", "predictor1", "predictor2"]], segment2["response"]
    )

    # Verify the coefficients are indeed different between segments
    assert not np.allclose(lr1.coef_, lr2.coef_, rtol=0.3), (
        "Coefficients should be different between segments"
    )


def test_check_data_column():
    """Test the check_data_column function through LinearRegressionCost."""
    # Create test data
    X = np.random.rand(100, 4)
    df = pd.DataFrame(X, columns=["col1", "col2", "col3", "col4"])

    # Test valid integer response column
    cost = LinearRegressionCost(response_col=2)
    cost.fit(X)
    assert cost._response_col_idx == 2

    # Test valid string response column with DataFrame
    cost = LinearRegressionCost(response_col="col3")
    cost.fit(df)
    assert cost._response_col_idx == 2

    # Test valid specific covariate columns (integers)
    cost = LinearRegressionCost(response_col=0, covariate_cols=[1, 3])
    cost.fit(X)
    assert cost._response_col_idx == 0
    assert cost._covariate_col_indices == [1, 3]

    # Test valid specific covariate columns (strings) with DataFrame
    cost = LinearRegressionCost(response_col="col1", covariate_cols=["col2", "col4"])
    cost.fit(df)
    assert cost._response_col_idx == 0
    assert cost._covariate_col_indices == [1, 3]

    # Test error cases - out of bounds index
    with pytest.raises(ValueError, match="Response column index.*must be between"):
        cost = LinearRegressionCost(response_col=10)  # Out of bounds
        cost.fit(X)

    # Test error cases - invalid column name
    with pytest.raises(ValueError, match="Response column.*not found among"):
        cost = LinearRegressionCost(response_col="invalid_column")
        cost.fit(df)

    # Test error cases - string column name with numpy array (no column names)
    with pytest.raises(ValueError, match="Response column must be an integer"):
        cost = LinearRegressionCost(response_col="col1")
        cost.fit(X)  # X is numpy array, not DataFrame

    # Test error cases - invalid covariate column
    with pytest.raises(ValueError, match="Covariate column.*not found among"):
        cost = LinearRegressionCost(response_col=0, covariate_cols=[1, "invalid"])
        cost.fit(df)

    # Test that all columns except response are used as covariates by default
    cost = LinearRegressionCost(response_col=2)
    cost.fit(X)
    assert cost._covariate_col_indices == [0, 1, 3]


def test_check_fixed_param_dimension_validation():
    """Test validation of parameter dimensions in _check_fixed_param."""
    # Create simple dataset
    X = np.random.rand(100, 3)

    # Test case with 2D array of wrong shape (width > 1)
    invalid_params_wide = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="Expected 2 coefficients"):
        cost = LinearRegressionCost(param=invalid_params_wide, response_col=0)
        cost.fit(X)

    # Test 3D array (completely invalid shape)
    invalid_params_3d = np.zeros((1, 2, 1))
    with pytest.raises(ValueError, match="Coefficients must have shape"):
        cost = LinearRegressionCost(param=invalid_params_3d, response_col=0)
        cost.fit(X)

    # Test row vector vs column vector (both should work)
    row_vector = np.array([0.1, 0.2])  # Shape (2,)
    col_vector = np.array([[0.1], [0.2]])  # Shape (2, 1)

    cost_row = LinearRegressionCost(param=row_vector, response_col=0)
    cost_row.fit(X)
    assert cost_row._coeffs.shape == (2, 1)

    cost_col = LinearRegressionCost(param=col_vector, response_col=0)
    cost_col.fit(X)
    assert cost_col._coeffs.shape == (2, 1)


def test_linear_regression_cost_underdetermined_system():
    """Test LinearRegressionCost on an underdetermined system."""
    # Create a dataset where we have more features than samples
    n_features = 3
    n_samples = 5  # Fewer samples than features -> underdetermined

    # Create predictors
    X_features = np.ones((n_samples, n_features))

    # Create response
    np.random.seed(42)
    y = np.random.rand(n_samples)

    lr_low_rank = LinearRegression(fit_intercept=False).fit(X_features, y)
    y_pred = lr_low_rank.predict(X_features)
    scikit_residual = np.square(y - y_pred).sum()

    (np_coeffs, empty_residuals, X_rank, _) = np.linalg.lstsq(X_features, y)
    assert X_rank < n_features, "Matrix should be rank-deficient"
    assert empty_residuals.size == 0, "Residuals should be empty"

    y_np_lstsq_pred = np.dot(X_features, np_coeffs.reshape(-1, 1))
    residuals_np_lstsq = np.square(y.reshape(-1, 1) - y_np_lstsq_pred).sum()

    # Stack response and features
    X_with_y = np.hstack((y.reshape(-1, 1), X_features))

    # Fit the cost with y as the response (first column)
    cost = LinearRegressionCost(response_col=0)
    cost.fit(X_with_y)

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([n_samples])
    costs = cost.evaluate(cuts=np.column_stack((starts, ends)))

    # For an overdetermined system, the residuals are not necessarily zero.
    assert np.isclose(costs[0, 0], scikit_residual), (
        "Cost should be close to scikit_residuals for an underdetermined system"
    )
    assert np.isclose(costs[0, 0], residuals_np_lstsq), (
        "Cost should close to numpy residuals for an underdetermined system"
    )
