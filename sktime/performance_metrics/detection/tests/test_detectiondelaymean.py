"""Tests for DetectionDelayMean metric."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection import DetectionDelayMean


def test_detection_delay_mean_basic():
    """Test basic functionality and edge cases."""
    metric = DetectionDelayMean()

    # Single true event - early detection (should be 0)
    y_true = pd.DataFrame({"ilocs": [100]})
    y_pred = pd.DataFrame({"ilocs": [90]})
    assert metric(y_true, y_pred) == 0.0

    # On-time detection
    assert metric(y_true, pd.DataFrame({"ilocs": [100]})) == 0.0

    # Delayed detection
    assert metric(y_true, pd.DataFrame({"ilocs": [110]})) == 10.0

    # Multiple events with greedy matching
    y_true = pd.DataFrame({"ilocs": [100, 200]})
    y_pred = pd.DataFrame({"ilocs": [105, 195]})
    score = metric(y_true, y_pred)
    assert score > 500  # one matched (delay 5), one unmatched → penalty

    # No true events
    assert metric(pd.DataFrame({"ilocs": []}), y_pred) == 0.0

    # No predictions (should apply penalty)
    assert metric(y_true, pd.DataFrame({"ilocs": []})) > 500


def test_detection_delay_mean_early_tolerance():
    """Test early_tolerance parameter."""
    metric = DetectionDelayMean(early_tolerance=5)

    y_true = pd.DataFrame({"ilocs": [100]})

    # Early by 3 → within tolerance → delay 0
    assert metric(y_true, pd.DataFrame({"ilocs": [97]})) == 0.0

    # Early by 6 → too early → penalty
    score = metric(y_true, pd.DataFrame({"ilocs": [94]}))
    assert score > 500

    # Multiple events
    y_true_multi = pd.DataFrame({"ilocs": [100, 200]})
    y_pred_multi = pd.DataFrame({"ilocs": [96, 205]})
    # 100 -> 96 (within tolerance → 0), 200 -> 205 (delay 5)
    score = metric(y_true_multi, y_pred_multi)
    assert score == 2.5


def test_detection_delay_mean_max_delay():
    """Test max_delay as cap and penalty."""
    metric = DetectionDelayMean(max_delay=50)

    y_true = pd.DataFrame({"ilocs": [100]})
    # Very late → capped at 50
    assert metric(y_true, pd.DataFrame({"ilocs": [200]})) == 50.0

    # No prediction → penalty = max_delay
    assert metric(y_true, pd.DataFrame({"ilocs": []})) == 50.0


def test_detection_delay_mean_combined_params():
    """Test early_tolerance + max_delay together."""
    metric = DetectionDelayMean(early_tolerance=5, max_delay=30)

    y_true = pd.DataFrame({"ilocs": [100, 200]})
    y_pred = pd.DataFrame({"ilocs": [96, 250]})

    # 100 -> 96 → 0, 200 -> 250 (50 → capped at 30)
    score = metric(y_true, y_pred)
    assert score == 15.0


def test_detection_delay_mean_get_test_params():
    """Test that get_test_params returns valid parameters."""
    params = DetectionDelayMean.get_test_params()
    assert len(params) >= 1
    for p in params:
        metric = DetectionDelayMean(**p)
        assert isinstance(metric, DetectionDelayMean)


@pytest.mark.parametrize(
    "y_true_ilocs, y_pred_ilocs, expected",
    [
        ([100], [110], 10.0),
        (
            [100],
            [95],
            0.0,
        ),  # early by 5 with tolerance=0 → still 0? No, default tolerance=0
        ([], [100], 0.0),
        ([100, 200], [105, 210], 7.5),  # delay 5 and 10
    ],
)
def test_detection_delay_mean_parametrized(y_true_ilocs, y_pred_ilocs, expected):
    """Parametrized test for various cases."""
    metric = DetectionDelayMean()
    y_true = pd.DataFrame({"ilocs": y_true_ilocs})
    y_pred = pd.DataFrame({"ilocs": y_pred_ilocs})
    result = metric(y_true, y_pred)
    assert abs(result - expected) < 1e-6
