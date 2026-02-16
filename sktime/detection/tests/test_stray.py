"""Tests for STRAY outlier estimator."""

__author__ = ["KatieBuc"]

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from sktime.detection.stray import STRAY
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(STRAY),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_default_1D():
    """Test with default parameters and 1D input array."""
    X = np.array(
        [
            -7.207066,
            -5.722571,
            -4.915559,
            -8.345698,
            -5.570875,
            -5.493944,
            -6.574740,
            -6.546632,
            -6.564452,
            -6.890038,
            -6.477193,
            -6.998386,
            -6.776254,
            -5.935541,
            -5.040506,
            0.000000,
            5.889715,
            5.488990,
            5.088805,
            5.162828,
            8.415835,
            6.134088,
            5.509314,
            5.559452,
            6.459589,
            5.306280,
            4.551795,
            6.574756,
            4.976344,
            5.984862,
            5.064051,
            7.102298,
        ]
    )

    y_expected = np.array(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
    )

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, 1))
    model = STRAY()
    y_actual = model.fit_transform(X)
    assert np.allclose(y_actual, y_expected)


@pytest.mark.skipif(
    not run_test_for_class(STRAY),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_default_2D():
    """Test with default parameters and 2D input array."""
    X = np.array(
        [
            [-1.207, -0.776, -0.694],
            [0.277, 0.064, -1.448],
            [1.084, 0.959, 0.575],
            [-2.346, -0.11, -1.024],
            [0.429, -0.511, -0.015],
            [0.506, -0.911, -0.936],
            [-0.575, -0.837, 1.102],
            [-0.547, 2.416, -0.476],
            [-0.564, 0.134, -0.709],
            [-0.89, -0.491, -0.501],
            [-0.477, -0.441, -1.629],
            [-0.998, 0.46, -1.168],
            [10.0, 12.0, 10.0],
            [3.0, 7.0, 10.0],
        ]
    )

    y_expected = np.array(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ]
    )

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model = STRAY()
    y_actual = model.fit_transform(X)
    assert np.allclose(y_actual, y_expected)


@pytest.mark.skipif(
    not run_test_for_class(STRAY),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_1D_score_with_na():
    """Test score with 1D input array with missing values."""
    X = np.array(
        [
            np.nan,
            -5.72257076,
            -4.91555882,
            -8.3456977,
            -5.57087531,
            0.0,
            6.50605589,
            5.42526004,
            5.45336814,
            5.435548,
            5.10996217,
        ]
    )

    y_scores_expected = np.array(
        [
            np.nan,
            0.17662069,
            0.23095851,
            0.17662069,
            0.18683466,
            0.33097498,
            0.07087969,
            0.02122967,
            0.02312225,
            0.02192238,
            0.02122967,
        ]
    )

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, 1))
    model = STRAY(k=3)
    fitted_model = model.fit(X)
    y_scores_actual = fitted_model.score_
    assert np.allclose(y_scores_actual, y_scores_expected, equal_nan=True)


@pytest.mark.skipif(
    not run_test_for_class(STRAY),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_1D_bool_with_na():
    """Test anomaly detection with 1D input array with missing values."""
    X = np.array(
        [
            np.nan,
            -5.72257076,
            -4.91555882,
            -8.3456977,
            -5.57087531,
            0.0,
            6.50605589,
            5.42526004,
            5.45336814,
            5.435548,
            5.10996217,
        ]
    )

    y_expected = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, 1))
    model = STRAY(k=3)
    fitted_model = model.fit(X)
    y_actual = fitted_model.y_
    assert np.allclose(y_actual, y_expected)


@pytest.mark.skipif(
    not run_test_for_class(STRAY),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_2D_score_with_na():
    """Test score with 2D input array with missing values."""
    X = np.array(
        [
            [-1.20706575, -0.57473996],
            [0.27742924, -0.54663186],
            [1.08444118, np.nan],
            [-2.3456977, -0.89003783],
            [0.42912469, -0.4771927],
            [0.50605589, -0.99838644],
        ]
    )

    y_scores_expected = np.array(
        [0.43612712, 0.43612712, np.nan, 0.69004259, 0.51240831, 0.85697804]
    )

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model = STRAY(k=2, size_threshold=4)
    fitted_model = model.fit(X)
    y_scores_actual = fitted_model.score_
    assert np.allclose(y_scores_actual, y_scores_expected, equal_nan=True)


@pytest.mark.skipif(
    not run_test_for_class(STRAY),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_2D_bool_with_na():
    """Test anomaly detection with 2D input array with missing values."""
    X = np.array(
        [
            [-1.20706575, -0.57473996],
            [0.27742924, -0.54663186],
            [1.08444118, np.nan],
            [-2.3456977, -0.89003783],
            [0.42912469, -0.4771927],
            [0.50605589, -0.99838644],
        ]
    )

    y_expected = np.array([0, 0, 0, 1, 1, 1])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model = STRAY(k=2, size_threshold=4)
    fitted_model = model.fit(X)
    y_actual = fitted_model.y_
    assert np.allclose(y_actual, y_expected)


@pytest.mark.skipif(
    not run_test_for_class(STRAY),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_2D_score_with_standardize():
    """Test score with 2D input array and median/IQR normalization."""
    X = np.array(
        [
            [-1.20706575, -0.57473996],
            [0.27742924, -0.54663186],
            [1.08444118, -0.5644520],
            [-2.3456977, -0.89003783],
            [0.42912469, -0.4771927],
            [0.50605589, -0.99838644],
        ]
    )

    y_scores_expected = np.array(
        [
            1.1274565,
            0.6139288,
            0.5982989,
            1.4866554,
            0.5982989,
            1.7245212,
        ]
    )

    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    model = STRAY(k=2, size_threshold=4)
    fitted_model = model.fit(X)
    y_scores_actual = fitted_model.score_
    assert np.allclose(y_scores_actual, y_scores_expected)
