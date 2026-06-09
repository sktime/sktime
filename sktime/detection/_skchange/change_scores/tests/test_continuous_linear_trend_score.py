"""Test cases for the ContinuousLinearTrendScore class and related functions."""

import datetime
import re

import numpy as np
import pandas as pd
import pytest

from sktime.detection._skchange.change_detectors import (
    MovingWindow,
    SeededBinarySegmentation,
)
from sktime.detection._skchange.change_scores import ContinuousLinearTrendScore
from sktime.detection._skchange.change_scores._continuous_linear_trend_score import (
    analytical_cont_piecewise_linear_trend_score,
    lin_reg_cont_piecewise_linear_trend_score,
)
from sktime.detection._skchange.datasets import (
    generate_continuous_piecewise_linear_signal,
)


def test_moving_window_single_changepoint():
    """Test MovingWindow with ContinuousLinearTrendScore on a single changepoint."""
    # Generate data with a single changepoint at position 100
    true_change_points = [100]
    slopes = [0.5, -0.5]  # Positive slope followed by negative slope
    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=200,
        noise_std=1.5,
        random_seed=42,
    )

    # Create detector with ContinuousLinearTrendScore
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(),
        bandwidth=30,
        penalty=25,  # Tuned for this specific test case
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df)

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Assert the changepoint is close to the true changepoint
    detected_cp = detected_cps.iloc[0, 0]
    cp_detection_margin = 3
    assert abs(detected_cp - true_change_points[0]) <= cp_detection_margin, (
        f"Detected {detected_cp}, expected close to {true_change_points[0]}"
    )


def test_moving_window_multiple_changepoints():
    """Test MovingWindow with ContinuousLinearTrendScore on multiple changepoints."""
    # Generate data with multiple changepoints
    true_change_points = [100, 200, 300]
    slopes = [0.5, -0.5, 0.4, -0.4]
    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=400,
        noise_std=1.5,
        random_seed=42,
    )

    # Create detector with ContinuousLinearTrendScore
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(), bandwidth=50, penalty=20
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df)

    # Assert we found the correct number of changepoints
    assert len(detected_cps) == len(true_change_points), (
        f"Expected {len(true_change_points)} changepoints, found {len(detected_cps)}"
    )

    # Assert the changepoints are close to the true changepoints
    cp_detection_margin = 3
    for i, cp in enumerate(detected_cps["ilocs"]):
        assert abs(cp - true_change_points[i]) <= cp_detection_margin, (
            f"Detected {cp}, expected close to {true_change_points[i]}"
        )


def test_seeded_binseg_single_changepoint():
    """Test SeededBinarySegmentation with ContinuousLinearTrendScore

    On a single changepoint.
    """
    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.1, -0.2]
    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=200,
        noise_std=1.5,
        random_seed=42,
    )

    # Create detector with ContinuousLinearTrendScore:
    detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(),
        penalty=25,
        selection_method="narrowest",
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df)

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Assert the changepoint is close to the true changepoint
    detected_cp = detected_cps.iloc[0, 0]
    cp_detection_margin = 2
    assert abs(detected_cp - true_change_points[0]) <= cp_detection_margin, (
        f"Detected {detected_cp}, expected close to {true_change_points[0]}"
    )


def test_seeded_binseg_multiple_changepoints():
    """Test SeededBinarySegmentation with ContinuousLinearTrendScore

    On multiple changepoints.
    """
    # Generate data with multiple changepoints
    true_change_points = [100, 200, 300]
    slopes = [0.1, -0.2, 0.15, -0.1]
    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=400,
        noise_std=2.0,
        random_seed=42,
    )

    # Create detector with ContinuousLinearTrendScore:
    detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(),
        selection_method="narrowest",
        penalty=45.0,  # ilocs: 101, 195, 302
        # selection_method="greedy",
        ## Low penalty that finds 3 changepoints when using greedy:
        # penalty=45,
        ## High penalty that still finds 3 changepoints when using greedy:
        # penalty=500,
    )

    # Fit and predict:
    detected_cps = detector.fit_predict(df)

    # Assert we found the correct number of changepoints
    assert len(detected_cps) == len(true_change_points), (
        f"Expected {len(true_change_points)} changepoints, found {len(detected_cps)}"
    )

    # Assert the changepoints are close to the true changepoints
    cp_detection_margin = 5
    for i, cp in enumerate(detected_cps["ilocs"]):
        detection_error = abs(cp - true_change_points[i])
        print("Detection error:", detection_error)
        assert detection_error <= cp_detection_margin, (
            f"Detected change at index {cp}, expected close to {true_change_points[i]}"
        )


def test_noise_sensitivity():
    """Test the sensitivity of both algorithms to different noise levels."""
    # Generate data with a single changepoint
    true_cps = [100]
    slopes = [0.1, -0.2]

    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
    max_deviations = []

    for noise_std in noise_levels:
        df = generate_continuous_piecewise_linear_signal(
            change_points=true_cps,
            slopes=slopes,
            n_samples=200,
            noise_std=noise_std,
            random_seed=42,
        )

        # Test Narrowest over threshold
        detector = SeededBinarySegmentation(
            ContinuousLinearTrendScore(), penalty=25, selection_method="narrowest"
        )

        cps = detector.fit_predict(df)

        if len(cps) == 1:
            deviation = abs(cps.iloc[0, 0] - true_cps[0])
            max_deviations.append(deviation)

    # Assert that for reasonable noise levels, we can detect the changepoint
    assert len(max_deviations) == len(noise_levels), (
        "Detection worked for low noise levels"
    )
    assert max(max_deviations) < 3, (
        f"Detection failed, with max deviation: {max(max_deviations)}"
    )
    # For lower noise levels, the detection should be more accurate
    assert all(np.diff(max_deviations) >= 0), (
        "Detection accuracy doesn't improve with lower noise"
    )


def test_multivariate_detection():
    """Test detection on multivariate continuous piecewise linear signals."""
    # Generate two different signals with the same changepoints
    change_points = [100, 200]
    slopes1 = [0.1, -0.2, 0.15]
    slopes2 = [0.05, 0.15, -0.1]

    df1 = generate_continuous_piecewise_linear_signal(
        change_points=change_points,
        slopes=slopes1,
        n_samples=300,
        noise_std=0.1,
        random_seed=42,
    )

    df2 = generate_continuous_piecewise_linear_signal(
        change_points=change_points,
        slopes=slopes2,
        n_samples=300,
        noise_std=0.1,
        random_seed=43,
    )

    # Combine into multivariate DataFrame
    df = pd.DataFrame({"signal1": df1["signal"], "signal2": df2["signal"]})

    # Test with SeededBinarySegmentation
    detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(),
        penalty=25,
        selection_method="greedy",
    )

    detected_cps = detector.fit_predict(df)

    # Assert we found the correct number of changepoints
    assert len(detected_cps) == len(change_points), (
        f"Expected {len(change_points)} changepoints, found {len(detected_cps)}"
    )

    # Assert the changepoints are close to the true changepoints
    cp_detection_margin = 3
    for i, cp in enumerate(detected_cps["ilocs"]):
        assert abs(cp - change_points[i]) <= cp_detection_margin, (
            f"Detected {cp}, expected close to {change_points[i]}"
        )


def test_irregular_time_sampling():
    """Test ContinuousLinearTrendScore with irregular time sampling."""
    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.1, -0.2]
    n_samples = 200

    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=n_samples,
        noise_std=1.0,
        random_seed=42,
    )
    df["sample_times"] = 50.0 + 2.0 * np.arange(n_samples)

    # Create irregular time sampling by selectively removing points
    # Keep all points around the changepoint for accurate detection
    np.random.seed(42)
    selection_mask = np.ones(n_samples, dtype=bool)

    # Remove ~30% of points away from the changepoint
    for region in [(0, 80), (90, 110), (120, 200)]:
        start, end = region
        region_len = end - start
        # Remove about 30% of points in each region
        to_remove = np.random.choice(
            np.arange(start, end), size=int(region_len * 0.3), replace=False
        )
        selection_mask[to_remove] = False

    # Apply the mask to create irregularly sampled data:
    irregular_df = df[selection_mask].copy()
    reverse_index_map = df.index[selection_mask]
    # Create a sample_times column that reflects the original indices
    true_change_point_time = df["sample_times"].iloc[true_change_points[0]]

    # Create detector with ContinuousLinearTrendScore with time_column
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(time_column="sample_times"),
        bandwidth=30,
        penalty=25,
    )

    # Fit and predict
    detected_cps = detector.fit_predict(irregular_df)

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Get the original index that corresponds to the detected changepoint
    detected_cp_idx = reverse_index_map[detected_cps.iloc[0, 0]]
    detected_cp_time = df["sample_times"].iloc[detected_cp_idx]

    # Assert the detected time is close to the true changepoint
    cp_detection_margin = 2  # Slightly larger margin for irregular sampling
    cp_time_detection_margin = 4.0
    assert abs(detected_cp_idx - true_change_points[0]) <= cp_detection_margin, (
        f"Detection index {detected_cp_idx}, expected close to {true_change_points[0]}"
    )

    # Assert the detected time is close to the true changepoint
    assert abs(detected_cp_time - true_change_point_time) <= cp_time_detection_margin, (
        f"Detected time {detected_cp_time}, expected close \
        to {true_change_point_time}"
    )

    # Test with SeededBinarySegmentation as well
    sbs_detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(time_column="sample_times"),
        selection_method="greedy",
        penalty=25,
    )

    # Fit and predict
    sbs_detected_cps = sbs_detector.fit_predict(irregular_df)

    # Assert we found 1 changepoint
    assert len(sbs_detected_cps) == 1, (
        f"Expected 1 changepoint, found {len(sbs_detected_cps)}"
    )

    # Get the original time that corresponds to the detected changepoint
    sbs_detected_cp_idx = reverse_index_map[sbs_detected_cps.iloc[0, 0]]
    sbs_detected_cp_time = df["sample_times"].iloc[sbs_detected_cp_idx]

    # Assert the detected time is close to the true changepoint
    assert abs(sbs_detected_cp_idx - true_change_points[0]) <= cp_detection_margin, (
        f"SBS detected at time {sbs_detected_cp_time}, expected close \
          to {true_change_points[0]}"
    )
    # Assert the detected time is close to the true changepoint
    assert (
        abs(sbs_detected_cp_time - true_change_point_time) <= cp_time_detection_margin
    ), (
        f"SBS detected time {sbs_detected_cp_time}, expected close \
          to {true_change_point_time}"
    )


def test_ignoring_irregular_time_sampling():
    """Test ContinuousLinearTrendScore when ignoring irregular time sampling."""
    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.1, -0.2]
    n_samples = 200

    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=n_samples,
        noise_std=1.0,
        random_seed=42,
    )
    # df["sample_times"] = 50.0 + 2.0 * np.arange(n_samples)

    # Create irregular time sampling by selectively removing points
    # Keep all points around the changepoint for accurate detection
    np.random.seed(42)
    selection_mask = np.ones(n_samples, dtype=bool)

    # Remove ~30% of points away from the changepoint
    for region in [(0, 80), (90, 110), (120, 200)]:
        start, end = region
        region_len = end - start
        # Remove about 30% of points in each region
        to_remove = np.random.choice(
            np.arange(start, end), size=int(region_len * 0.4), replace=False
        )
        selection_mask[to_remove] = False

    # Apply the mask to create irregularly sampled data:
    irregular_df = df[selection_mask].copy()
    reverse_index_map = df.index[selection_mask]

    # Closest index to the true changepoint in the original data:
    true_cp_index = np.where(selection_mask[: true_change_points[0] + 1])[0][-1]

    # Create detector with ContinuousLinearTrendScore WITHOUT time_column
    mw_detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(),  # No time_column provided
        bandwidth=30,
        penalty=25,
    )

    # Fit and predict
    mw_detected_cps = mw_detector.fit_predict(irregular_df)

    # Assert we found 1 changepoint
    assert len(mw_detected_cps) == 1, (
        f"Expected 1 changepoint, found {len(mw_detected_cps)}"
    )

    # Get the detected changepoint index in the irregular data
    mw_detected_cp_idx = reverse_index_map[mw_detected_cps.iloc[0, 0]]

    # Without time information, the detected changepoint index should be different
    # from the one we would expect with time information
    lower_cp_detection_margin = 2
    upper_cp_detection_margin = 12
    assert (
        lower_cp_detection_margin
        < abs(mw_detected_cp_idx - true_cp_index)
        < upper_cp_detection_margin
    ), f"Detection could fail when ignoring irregular sampling: {mw_detected_cp_idx}"

    # Test with SeededBinarySegmentation as well
    sbs_detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(),  # No time_column provided
        selection_method="narrowest",
        penalty=25,
    )

    # Fit and predict
    sbs_detected_cps = sbs_detector.fit_predict(irregular_df)

    # Assert we found 1 changepoint
    assert len(sbs_detected_cps) == 1, (
        f"Expected 1 changepoint, found {len(sbs_detected_cps)}"
    )


def fit_indexed_linear_trend(xs: np.ndarray) -> tuple[float, float]:
    """Calculate the optimal linear trend for a given array.

    Assuming the time steps are [0, 1, 2, ..., n-1], we can optimize the calculation
    of the least squares intercept and slope.

    Parameters
    ----------
    xs : np.ndarray
        1D array of data points

    Returns
    -------
    tuple
        (slope, intercept) of the best-fit line
    """
    n_samples = len(xs)

    # For evenly spaced time steps [0, 1, 2, ..., n-1],
    # the mean time step is (n-1)/2.
    mean_t = (n_samples - 1) / 2.0

    # Optimized calculation for denominator:
    # sum of (t - mean_t)^2 = n*(n^2-1)/12
    denominator = n_samples * (n_samples * n_samples - 1) / 12.0

    # Calculate numerator: sum((t-mean_t)*(x-mean_x))
    # numerator = np.sum((np.arange(n) - mean_t) * (xs - mean_x))
    mean_x = np.mean(xs)
    numerator = 0.0
    for i in range(n_samples):
        numerator += (i - mean_t) * (xs[i] - mean_x)

    slope = numerator / denominator
    intercept = mean_x - slope * mean_t

    return intercept, slope


def linear_trend_score(
    starts: np.ndarray, splits: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Evaluate the continuous linear trend cost.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the first intervals (inclusive).
    splits : np.ndarray
        Split indices between the intervals (contained in second interval).
    ends : np.ndarray
        End indices of the second intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    ### NOTE: Assume 'time' is index of the data. i.e. time = 0, 1, 2, ..., len(X)-1
    ###       This assumption could be changed later.
    n_intervals = len(starts)
    n_columns = X.shape[1]
    scores = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, split, end = starts[i], splits[i], ends[i]
        split_interval_trend_data = np.zeros((end - start, 3))
        split_interval_trend_data[:, 0] = 1.0  # Intercept
        # Whole interval slope:
        split_interval_trend_data[:, 1] = np.arange(end - start)  # Time steps

        # Change in slope from the split point:
        # trend data index starts at 0 from 'start'.
        # Continuous at the first point of the second interal, [split, end - 1]:
        split_interval_trend_data[(split - start) :, 2] = np.arange(end - split)

        # Calculate the slope and intercept for the whole interval:
        split_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data, X[start:end, :]
        )
        split_interval_squared_residuals = split_interval_linreg_res[1]

        # By only regressing onto the first two columns, we can calculate the cost
        # without allowing for a change in slope at the split point.
        joint_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data[:, [0, 1]], X[start:end, :]
        )
        joint_interval_squared_residuals = joint_interval_linreg_res[1]

        scores[i, :] = (
            joint_interval_squared_residuals - split_interval_squared_residuals
        )

    return scores


def test_linear_trend_score():
    # Straight linear trend test data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=float)
    times = np.arange(1, X.shape[0] + 1)
    starts = np.array([0])
    splits = np.array([2])
    ends = np.array([5])

    # Expected output
    expected_scores = np.array([[0.0, 0.0]])

    # Call the function
    scores = lin_reg_cont_piecewise_linear_trend_score(
        starts, splits, ends, X, times=times
    )

    # Check if the output matches the expected output
    assert np.allclose(scores, expected_scores), (
        f"Expected {expected_scores}, got {scores}"
    )


def test_continuous_linear_trend_score_function():
    """Test that continuous_linear_trend_score function works as expected."""
    # Test data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=float)
    starts = np.array([0])
    splits = np.array([2])
    ends = np.array([5])
    times = np.arange(X.shape[0])  # Default times [0, 1, 2, 3, 4]

    # Calculate scores using both methods
    lin_reg_score_values = lin_reg_cont_piecewise_linear_trend_score(
        starts, splits, ends, X, times
    )
    analytical_score_values = analytical_cont_piecewise_linear_trend_score(
        starts, splits, ends, X
    )

    # Check if all methods give the same result
    assert np.allclose(lin_reg_score_values, analytical_score_values)


def test_linear_trend_score_with_different_times():
    """Test that linear_trend_score works with custom times."""
    # Test data
    np.random.seed(0)  # For reproducibility
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=float)
    X += np.random.normal(0, 0.1, X.shape)  # Add some noise
    starts = np.array([0])
    splits = np.array([2])
    ends = np.array([5])

    # Default times [0, 1, 2, 3, 4]
    default_times = np.arange(X.shape[0])
    # Custom evenly spaced times
    custom_times = np.array([0, 2, 4, 6, 8])

    # Calculate scores
    scores_default = lin_reg_cont_piecewise_linear_trend_score(
        starts, splits, ends, X, default_times
    )
    scores_custom = lin_reg_cont_piecewise_linear_trend_score(
        starts, splits, ends, X, custom_times
    )

    # Since the times are evenly spaced but with different scales,
    # the scores should be the same
    assert np.allclose(scores_default, scores_custom)


def test_multiple_segments():
    """Test with multiple segments to evaluate."""
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]], dtype=float)
    # Add some noise:
    # np.random.seed(0)  # For reproducibility
    X += np.random.normal(0, 1.0, X.shape)  # Add some noise
    starts = np.array([0, 2])
    splits = np.array([2, 5])
    ends = np.array([5, 7])
    times = np.arange(X.shape[0])

    # Calculate scores using both methods
    scores_func = analytical_cont_piecewise_linear_trend_score(starts, splits, ends, X)
    scores_with_times = lin_reg_cont_piecewise_linear_trend_score(
        starts, splits, ends, X, times
    )

    # Check if both methods give the same result
    assert np.allclose(scores_func, scores_with_times)


def test_datetime_indices():
    """Test ContinuousLinearTrendScore with datetime indices."""

    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.5, -0.5]
    n_samples = 200

    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=n_samples,
        noise_std=1.0,
        random_seed=42,
    )

    # Create datetime index starting from today
    start_date = datetime.datetime.now()
    datetime_index = [start_date + datetime.timedelta(days=i) for i in range(n_samples)]
    df["datetime"] = datetime_index

    ## Sane way of creating a numeric time column:
    df["time_elapsed"] = df["datetime"] - df.loc[0, "datetime"]
    df["num_timestamp"] = (
        df["time_elapsed"].dt.total_seconds() / 3600.0
    )  # Convert to hours

    # Create detector with ContinuousLinearTrendScore with datetime column
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(time_column="num_timestamp"),
        bandwidth=30,
        penalty=25,
    )

    # Fit and predict:
    detected_cps = detector.fit_predict(df.drop(columns=["datetime", "time_elapsed"]))

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Assert the changepoint is close to the true changepoint
    detected_cp = detected_cps.iloc[0, 0]
    cp_detection_margin = 3
    assert abs(detected_cp - true_change_points[0]) <= cp_detection_margin, (
        f"Detected {detected_cp}, expected close to {true_change_points[0]}"
    )


def test_numpy_datetime64():
    """Test ContinuousLinearTrendScore with numpy datetime64 type."""
    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.1, -0.2]
    n_samples = 200

    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=n_samples,
        noise_std=1.0,
        random_seed=42,
    )

    # Create numpy datetime64 column
    start_date = np.datetime64("2023-01-01")
    dt_array = np.array([start_date + np.timedelta64(i, "h") for i in range(n_samples)])
    df["np_datetime"] = dt_array

    # Create a numeric timestamp:
    df["num_timestamp"] = (
        df["np_datetime"] - df["np_datetime"][0]
    ).dt.total_seconds() / (60.0 * 60.0)  # Convert to hours

    # Create detector with ContinuousLinearTrendScore with datetime column:
    detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(time_column="num_timestamp"),
        penalty=25,
        selection_method="narrowest",
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df.drop(columns=["np_datetime"]))

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Assert the changepoint is close to the true changepoint
    detected_cp = detected_cps.iloc[0, 0]
    cp_detection_margin = 3
    assert abs(detected_cp - true_change_points[0]) <= cp_detection_margin, (
        f"Detected {detected_cp}, expected close to {true_change_points[0]}"
    )


def test_pandas_timestamp():
    """Test ContinuousLinearTrendScore with pandas Timestamp type."""
    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.1, -0.2]
    n_samples = 200

    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=n_samples,
        noise_std=1.0,
        random_seed=42,
    )

    # Create pandas Timestamp column
    start_date = pd.Timestamp("2023-01-01")
    pd_timestamps = [start_date + pd.Timedelta(days=i) for i in range(n_samples)]
    df["pd_timestamp"] = pd_timestamps

    # Create detector with ContinuousLinearTrendScore with datetime column
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(time_column="pd_timestamp"),
        bandwidth=30,
        penalty=25,
    )

    with pytest.raises(
        TypeError,
        match=re.escape(
            "float() argument must be a string or a real number, not 'Timestamp'"
        ),
    ):
        # Fit and predict should raise an error:
        _ = detector.fit_predict(df)


def test_date_objects():
    """Test ContinuousLinearTrendScore with Python date objects."""

    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.5, -0.5]
    n_samples = 200

    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=n_samples,
        noise_std=1.0,
        random_seed=42,
    )

    # Create Python date objects
    start_date = datetime.date(2023, 1, 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(n_samples)]
    # Call 'convert_dtypes' to make sure the date column is of 'dtype' "date":
    df["date"] = pd.to_datetime(pd.Series(dates))
    # Create a numeric timestamp:
    df["days_since_first"] = (
        df["date"] - df.loc[0, "date"]
    ).dt.total_seconds() / pd.Timedelta(days=1).total_seconds()  # Convert to hours

    # Create detector with ContinuousLinearTrendScore with date column
    detector = SeededBinarySegmentation(
        change_score=ContinuousLinearTrendScore(time_column="days_since_first"),
        penalty=25,
        selection_method="narrowest",
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df.drop(columns=["date"]))

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Assert the changepoint is close to the true changepoint
    detected_cp = detected_cps.iloc[0, 0]
    cp_detection_margin = 3
    assert abs(detected_cp - true_change_points[0]) <= cp_detection_margin, (
        f"Detected {detected_cp}, expected close to {true_change_points[0]}"
    )


def test_non_uniform_datetime_sampling():
    """Test ContinuousLinearTrendScore with non-uniform datetime sampling."""

    # Generate data with a single changepoint
    true_change_points = [100]
    slopes = [0.5, -0.5]
    n_samples = 200

    df = generate_continuous_piecewise_linear_signal(
        change_points=true_change_points,
        slopes=slopes,
        n_samples=n_samples,
        noise_std=1.0,
        random_seed=42,
    )

    # Create non-uniform datetime sampling
    start_date = datetime.datetime(2023, 1, 1)
    np.random.seed(42)
    # Add random hours (0-23) to each day
    random_hours = np.random.randint(0, 24, n_samples)
    datetimes = [
        start_date + datetime.timedelta(days=int(i), hours=int(h))
        for i, h in enumerate(random_hours)
    ]
    df["datetime"] = pd.to_datetime(pd.Series(datetimes))
    df["num_timestamp"] = (
        df["datetime"] - df.loc[0, "datetime"]
    ).dt.total_seconds() / pd.Timedelta(days=1).total_seconds()
    # Note the timestamp at the true changepoint for later comparison
    true_cp_time = df["datetime"].iloc[true_change_points[0]]

    # Create detector with ContinuousLinearTrendScore with datetime column
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(time_column="num_timestamp"),
        bandwidth=30,
        penalty=25,
    )

    # Fit and predict
    detected_cps = detector.fit_predict(df.drop(columns=["datetime"]))

    # Assert we found 1 changepoint
    assert len(detected_cps) == 1, f"Expected 1 changepoint, found {len(detected_cps)}"

    # Assert the changepoint is close to the true changepoint
    detected_cp = detected_cps.iloc[0, 0]
    cp_detection_margin = 3
    assert abs(detected_cp - true_change_points[0]) <= cp_detection_margin, (
        f"Detected {detected_cp}, expected close to {true_change_points[0]}"
    )

    # Also check that the datetime is close
    detected_time = df["datetime"].iloc[detected_cp]
    time_diff = abs((detected_time - true_cp_time).total_seconds())
    max_allowed_diff = pd.Timedelta(days=4).total_seconds()
    assert time_diff <= max_allowed_diff, (
        f"Detected time {detected_time} too far from true time {true_cp_time}"
    )


def test_large_timestamp_values():
    """Test ContinuousLinearTrendScore with extremely large timestamp values."""
    # Generate test data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]], dtype=float)
    starts = np.array([0])
    splits = np.array([2])
    ends = np.array([5])

    # Create extremely large timestamps (e.g., microseconds since epoch)
    large_times = 1.676e13 + np.arange(X.shape[0]) * 1e9  # Large timestamp values

    # Calculate scores using the linear regression method
    scores = lin_reg_cont_piecewise_linear_trend_score(
        starts, splits, ends, X, large_times
    )

    # Check if scores contain NaN values
    assert np.any(np.isnan(scores)), "Expected NaN scores with large timestamp values"

    # Create a detector with a time column
    df = pd.DataFrame({"signal1": X[:, 0], "signal2": X[:, 1], "time": large_times})

    # Test graceful handling in MovingWindow
    detector = MovingWindow(
        change_score=ContinuousLinearTrendScore(time_column="time"),
        bandwidth=3,
        penalty=1.0,  # Low penalty to encourage changepoint detection
    )

    # The detector should not crash, even with extreme timestamp values
    try:
        detected_cps = detector.fit_predict(df)
        # If we get here, it didn't crash
        assert True, "MovingWindow handled large timestamps without crashing"
    except np.linalg.LinAlgError:
        # Expected behavior - singular matrix in least squares
        assert True, "Expected LinAlgError with large timestamps"
    except Exception as e:
        pytest.fail(f"Unexpected error type: {type(e)}, message: {str(e)}")

    # Alternative approach: scale down the timestamps
    normalized_times = (large_times - large_times[0]) / 1e9

    scores_normalized = lin_reg_cont_piecewise_linear_trend_score(
        starts, splits, ends, X, normalized_times
    )

    # Normalized timestamps should work properly and not produce NaN
    assert not np.any(np.isnan(scores_normalized)), (
        "Normalized timestamps should not produce NaN"
    )

    # Check that the detector works with normalized timestamps
    df["normalized_time"] = normalized_times
    detector_normalized = MovingWindow(
        change_score=ContinuousLinearTrendScore(time_column="normalized_time"),
        bandwidth=3,
        penalty=1.0,
    )

    # This should now work without numerical issues
    detected_cps = detector_normalized.fit_predict(df)
    assert isinstance(detected_cps, pd.DataFrame), (
        "Should return valid changepoints with normalized timestamps"
    )


def test_three_point_segment():
    """Test with a three-point segment."""
    X = np.array([[1, 2], [2, 3], [3, 4]], dtype=float)
    # Add some noise:
    np.random.seed(0)  # For reproducibility
    X += np.random.normal(0, 1.0, X.shape)  # Add some noise
    starts = np.array([0])
    splits = np.array([1])
    ends = np.array([3])
    times = np.arange(X.shape[0])

    # Calculate scores using both methods
    scores_func = analytical_cont_piecewise_linear_trend_score(starts, splits, ends, X)
    scores_with_times = lin_reg_cont_piecewise_linear_trend_score(
        starts, splits, ends, X, times
    )
    assert np.allclose(scores_func, scores_with_times), (
        f"Expected {scores_func}, got {scores_with_times}"
    )
