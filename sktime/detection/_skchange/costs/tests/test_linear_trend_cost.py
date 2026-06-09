"""Tests for LinearTrendCost class."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from sktime.detection._skchange.anomaly_detectors import CAPA
from sktime.detection._skchange.anomaly_scores import to_saving
from sktime.detection._skchange.change_detectors import PELT
from sktime.detection._skchange.costs import LinearTrendCost
from sktime.detection._skchange.costs._linear_trend_cost import (
    fit_indexed_linear_trend,
    fit_linear_trend,
)
from sktime.detection._skchange.penalties import make_linear_chi2_penalty


def test_linear_trend_cost_init():
    """Test initialization of LinearTrendCost."""
    # Default parameters
    cost = LinearTrendCost()
    assert cost.param is None
    assert cost.time_column is None

    # With fixed parameters for a single column
    cost = LinearTrendCost(param=np.array([2.0, 1.0]))  # slope=2, intercept=1
    assert cost.param is not None
    assert cost.time_column is None

    # With time column specified
    cost = LinearTrendCost(time_column="timestamp")
    assert cost.param is None
    assert cost.time_column == "timestamp"


def test_linear_trend_cost_fit():
    """Test fitting of LinearTrendCost."""
    # Create simple dataset
    X = np.random.rand(100, 3)

    # Default fit without time column
    cost = LinearTrendCost()
    cost.fit(X)
    assert cost.is_fitted
    assert cost._time_stamps is None

    # With time column as integer index
    test_time_column = 1
    cost = LinearTrendCost(time_column=test_time_column)
    cost.fit(X)
    assert cost.is_fitted
    assert_allclose(cost._time_stamps, X[:, test_time_column] - X[0, test_time_column])

    # With time column as string with DataFrame:
    df = pd.DataFrame(X, columns=["time", "value1", "value2"])
    cost = LinearTrendCost(time_column="time")
    cost.fit(df)
    assert cost.is_fitted
    assert_allclose(cost._time_stamps, df["time"].values - df["time"].values[0])

    # With fixed parameters:
    fixed_params = np.array([[1.0, 0.5], [2.0, 1.0], [0.5, 2.0]])
    cost = LinearTrendCost(param=fixed_params)
    cost.fit(X)
    assert cost.is_fitted
    assert cost._trend_params is not None
    assert_allclose(cost._trend_params, fixed_params)
    # When no time column is provided but params are fixed,
    # should create index time stamp sequence.
    assert_allclose(cost._time_stamps, np.arange(X.shape[0]))


def test_linear_trend_cost_param_validation():
    """Test parameter validation in LinearTrendCost."""
    X = np.random.rand(100, 3)

    # Valid parameter shape for 3 columns: (3, 2)
    valid_params = np.array([[1.0, 0.5], [2.0, 1.0], [0.5, 2.0]])
    cost = LinearTrendCost(param=valid_params)
    cost.fit(X)
    assert_allclose(cost._trend_params, valid_params)

    # Valid parameters for a single column provided as 1D array
    X_single = np.random.rand(100, 1)
    valid_params_1d = np.array([1.0, 0.5])  # slope, intercept
    cost = LinearTrendCost(param=valid_params_1d)
    cost.fit(X_single)
    assert_allclose(cost._trend_params, valid_params_1d.reshape(-1, 2))

    # Invalid parameter count
    invalid_params_count = np.array(
        [[1.0, 0.5], [2.0, 1.0]]
    )  # Only 2 rows for 3 columns
    with pytest.raises(ValueError, match="Expected .* parameters"):
        cost = LinearTrendCost(param=invalid_params_count)
        cost.fit(X)

    # Invalid parameter shape: (2, 3)
    invalid_params_shape = np.array([[1.0, 0.5, 0.3], [2.0, 1.0, 0.7]])
    with pytest.raises(ValueError, match="Fixed parameters must convertible to shape"):
        cost = LinearTrendCost(param=invalid_params_shape)
        cost.fit(X)


def test_fit_linear_trend():
    """Test the fit_linear_trend and fit_indexed_linear_trend functions."""
    # Create a simple linear trend
    np.random.seed(42)
    n_samples = 100
    true_slope = 2.5
    true_intercept = 1.0
    time_steps = np.linspace(0, 10, n_samples)
    values = (
        true_intercept + true_slope * time_steps + np.random.normal(0, 0.5, n_samples)
    )

    # Test fit_linear_trend:
    slope, intercept = fit_linear_trend(time_steps, values)
    assert np.isclose(slope, true_slope, rtol=0.1), (
        f"Expected slope ~{true_slope}, got {slope}"
    )
    assert np.isclose(intercept, true_intercept, rtol=0.1), (
        f"Expected intercept ~{true_intercept}, got {intercept}"
    )

    # Create evenly spaced data for fit_indexed_linear_trend:
    evenly_spaced_values = (
        true_intercept
        + true_slope * np.arange(n_samples)
        + np.random.normal(0, 0.5, n_samples)
    )
    slope_idx, intercept_idx = fit_indexed_linear_trend(evenly_spaced_values)

    # Test fit_indexed_linear_trend:
    assert np.isclose(slope_idx, true_slope, rtol=0.1), (
        f"Expected slope ~{true_slope}, got {slope_idx}"
    )
    assert np.isclose(intercept_idx, true_intercept, rtol=0.1), (
        f"Expected intercept ~{true_intercept}, got {intercept_idx}"
    )


def test_linear_trend_cost_evaluate():
    """Test evaluation of LinearTrendCost."""
    # Create data with a linear trend
    np.random.seed(52)
    n_samples = 100
    slope1, intercept1 = 2.0, 1.0
    slope2, intercept2 = -1.0, 5.0
    time_steps = np.linspace(0, 10, n_samples)

    # Create two columns with different trends
    y1 = intercept1 + slope1 * time_steps + np.random.normal(0, 0.3, n_samples)
    y2 = intercept2 + slope2 * time_steps + np.random.normal(0, 0.3, n_samples)

    X = np.column_stack([time_steps, y1, y2])

    # Fit the cost with time column 0
    cost = LinearTrendCost(time_column=0)
    cost.fit(X)

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([n_samples])
    costs = cost.evaluate(np.column_stack((starts, ends)))

    # Costs should be close to the noise variance:
    assert costs[0, 0] < 10.0  # Column 1 (y1)
    assert costs[0, 1] < 10.0  # Column 2 (y2)

    # Now evaluate on multiple intervals
    starts = np.array([0, n_samples // 2])
    ends = np.array([n_samples // 2, n_samples])
    costs_multi = cost.evaluate(np.column_stack((starts, ends)))

    # Sum of costs from both intervals should be less than cost
    # from entire interval.
    # (because we fit separate trends to each segment)
    assert costs_multi[0, 0] + costs_multi[1, 0] < costs[0, 0]
    assert costs_multi[0, 1] + costs_multi[1, 1] < costs[0, 1]


def test_linear_trend_cost_fixed_params():
    """Test evaluation with fixed parameters."""
    # Create data with a linear trend
    np.random.seed(21)
    n_samples = 100
    slope1, intercept1 = 2.0, 1.0
    time_steps = np.linspace(0, 10, n_samples)

    # Create values with some noise
    values = intercept1 + slope1 * time_steps + np.random.normal(0, 0.3, n_samples)

    X = np.column_stack([time_steps, values])

    # Fit with correct fixed parameters
    fixed_params = np.array([slope1, intercept1])
    cost_correct = LinearTrendCost(param=fixed_params, time_column=0)
    cost_correct.fit(X)

    # Fit with incorrect fixed parameters
    wrong_params = np.array([[slope1 * 1.5, intercept1 * 0.8]])
    cost_wrong = LinearTrendCost(param=wrong_params, time_column=0)
    cost_wrong.fit(X)

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([n_samples])
    cuts = np.column_stack((starts, ends))

    costs_correct = cost_correct.evaluate(cuts)
    costs_wrong = cost_wrong.evaluate(cuts)

    # Correct parameters should give lower cost than wrong parameters
    assert costs_correct[0, 0] < costs_wrong[0, 0]

    # Verify that min_size is 3 with fixed params and 3 without.
    assert cost_correct.min_size == 3

    cost_optim = LinearTrendCost(time_column=0)
    cost_optim.fit(X)
    assert cost_optim.min_size == 3


def test_linear_trend_cost_with_pelt():
    """Test LinearTrendCost with PELT algorithm on data with trend changes."""
    # Create data with two different linear trends
    n_samples = 200
    np.random.seed(42)

    # Time steps
    time_steps = np.linspace(0, 20, n_samples)

    # First segment: y = 2*t + 1 + noise
    y1 = 2.0 * time_steps[:100] + 1.0 + np.random.normal(0, 0.5, 100)

    # Second segment: y = -1*t + 15 + noise (change in both slope and intercept)
    y2 = -1.0 * time_steps[100:] + 15.0 + np.random.normal(0, 0.5, 100)

    # Combine
    y = np.concatenate([y1, y2])
    X = np.column_stack([time_steps, y])

    # Fit PELT with LinearTrendCost
    cost = LinearTrendCost(time_column=0)
    pelt = PELT(cost=cost, min_segment_length=10, penalty=5.0)
    result = pelt.fit_predict(X)

    # Should detect the changepoint at around index 100
    assert len(result) == 1, "Expected one changepoint"

    # Get the changepoint closest to the true one
    cp_idx = result["ilocs"][0]
    # Allow some tolerance
    assert abs(cp_idx - 100) <= 1, (
        f"Detected changepoint {cp_idx} too far from actual (100)"
    )

    # Create a DataFrame with named columns for testing string column names
    df = pd.DataFrame({"time": time_steps, "value": y})

    # Test with DataFrame and column names
    cost_df = LinearTrendCost(time_column="time")
    pelt_df = PELT(cost=cost_df, min_segment_length=10, penalty=5.0)
    result_df = pelt_df.fit_predict(df)

    # Should get similar results
    cp_idx_df = result_df["ilocs"][0]
    assert abs(cp_idx_df - 100) <= 1, (
        f"Detected changepoint with DataFrame {cp_idx_df} too far from actual (100)"
    )


def test_linear_trend_cost_default_time():
    """Test LinearTrendCost using default sequential time indices."""
    # Create data with a linear trend y = a*index + b
    np.random.seed(723)
    n_samples = 100
    slope, intercept = 2.0, 1.0

    # Create values using the array indices as time steps (0, 1, 2, ...)
    values = (
        intercept + slope * np.arange(n_samples) + np.random.normal(0, 0.3, n_samples)
    )

    # Reshape to column vector
    X = values.reshape(-1, 1)

    # Fit the cost without specifying time column
    cost = LinearTrendCost()
    cost.fit(X)

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([n_samples])
    costs = cost.evaluate(np.column_stack((starts, ends)))

    # The cost should be low, close to the noise variance
    assert costs[0, 0] < 10.0, "Cost should be low for simple linear trend."

    # Split the data and evaluate on two halves
    starts = np.array([0, n_samples // 2])
    ends = np.array([n_samples // 2, n_samples])
    costs_split = cost.evaluate(np.column_stack((starts, ends)))

    # Sum of costs from split segments should be lower than full interval
    # (or very close, since the trend is consistent)
    assert costs_split[0, 0] + costs_split[1, 0] <= costs[0, 0], (
        "Costs should be lower when split into segments"
    )


def test_get_model_size():
    """Test get_model_size method."""
    cost = LinearTrendCost()
    # For each column we need 2 parameters (slope and intercept)
    assert cost.get_model_size(3) == 6  # 3 columns * 2 params
    assert cost.get_model_size(1) == 2  # 1 column * 2 params


def test_indexed_trend_vs_explicit_time():
    """Test that using indexed time and explicit time gives similar results."""
    n_samples = 100
    slope, intercept = 2.0, 1.0

    # Create values with explicit time
    time_steps = np.arange(n_samples)
    values = intercept + slope * time_steps + np.random.normal(0, 0.1, n_samples)

    X_with_time = np.column_stack([time_steps, values])
    X_values_only = values.reshape(-1, 1)

    # Fit with explicit time column
    cost_explicit = LinearTrendCost(time_column=0)
    cost_explicit.fit(X_with_time)

    # Fit with indexed time (default)
    cost_indexed = LinearTrendCost()
    cost_indexed.fit(X_values_only)

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([n_samples])
    cuts = np.column_stack((starts, ends))

    costs_explicit = cost_explicit.evaluate(cuts)
    costs_indexed = cost_indexed.evaluate(cuts)

    # Both methods should give very similar results
    assert np.isclose(costs_explicit[0, 0], costs_indexed[0, 0], rtol=1e-6)


def test_linear_trend_cost_as_saving():
    """Test using LinearTrendCost as a Saving within MVCAPA detector."""

    # Create data with two trend anomalies
    n_samples = 300
    np.random.seed(25)

    # Normal data: constant with slight noise
    normal_data = np.random.normal(0, 0.2, (n_samples, 2))

    # First anomaly: linear trend in first column only (index 80-100)
    anomaly_idx1 = slice(80, 100)
    t1 = np.arange(anomaly_idx1.stop - anomaly_idx1.start)
    normal_data[anomaly_idx1, 0] = 0.5 * t1 + np.random.normal(0, 0.2, len(t1))

    # Second anomaly: linear trend in both columns (index 200-230)
    anomaly_idx2 = slice(200, 230)
    t2 = np.arange(anomaly_idx2.stop - anomaly_idx2.start)
    normal_data[anomaly_idx2, 0] = 0.7 * t2 + np.random.normal(0, 0.2, len(t2))
    normal_data[anomaly_idx2, 1] = -0.5 * t2 + np.random.normal(0, 0.2, len(t2))

    # Create DataFrame
    df = pd.DataFrame(normal_data, columns=["var1", "var2"])

    # Create LinearTrendCost with zero slope and intercept as baseline
    baseline_cost = LinearTrendCost(param=np.array([[0.0, 0.0], [0.0, 0.0]]))

    # Convert to Saving
    trend_saving = to_saving(baseline_cost)

    # Create MVCAPA detector with LinearTrendCost as Saving
    n = df.shape[0]
    p = df.shape[1]
    penalty = make_linear_chi2_penalty(trend_saving.get_model_size(p), n, p)
    detector = CAPA(
        segment_saving=trend_saving,
        segment_penalty=penalty,
        min_segment_length=5,
        max_segment_length=50,
        find_affected_components=True,
    )

    # Detect anomalies
    anomalies = detector.fit_predict(df)

    # Check that both anomalies were detected
    assert len(anomalies) >= 2, "Should detect at least two anomalies"

    # Check if the anomalies are close to the ground truth
    detected_intervals = [
        (interval.left, interval.right) for interval in anomalies["ilocs"]
    ]

    # Check if any detected interval overlaps with the first anomaly
    first_anomaly_detected = any(
        (start <= anomaly_idx1.start and end >= anomaly_idx1.stop)
        # or (start >= anomaly_idx1.start and start < anomaly_idx1.stop)
        # or (end > anomaly_idx1.start and end <= anomaly_idx1.stop)
        for start, end in detected_intervals
    )

    # Check if any detected interval overlaps with the second anomaly
    second_anomaly_detected = any(
        (start <= anomaly_idx2.start and end >= anomaly_idx2.stop)
        # or (start >= anomaly_idx2.start and start < anomaly_idx2.stop)
        # or (end > anomaly_idx2.start and end <= anomaly_idx2.stop)
        for start, end in detected_intervals
    )

    assert first_anomaly_detected, "First anomaly (index 80-100) was not detected"
    assert second_anomaly_detected, "Second anomaly (index 200-230) was not detected"

    # Check that the affected columns are correctly identified
    # First anomaly should only affect the first column
    first_anomaly_cols = next(
        (
            cols
            for (start, end), cols in zip(detected_intervals, anomalies["icolumns"])
            if start <= anomaly_idx1.stop and end >= anomaly_idx1.start
        ),
        None,
    )

    assert len(first_anomaly_cols) == 1 and first_anomaly_cols[0] == 0, (
        "First column should be affected in first anomaly"
    )

    # Second anomaly should affect both columns
    second_anomaly_cols = next(
        (
            cols
            for (start, end), cols in zip(detected_intervals, anomalies["icolumns"])
            if start <= anomaly_idx2.stop and end >= anomaly_idx2.start
        ),
        None,
    )

    assert {0, 1} == set(second_anomaly_cols), (
        "Both columns should be affected in second anomaly"
    )
