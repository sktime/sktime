"""Tests for DetectionDelayMean metric."""

import pandas as pd

from sktime.performance_metrics.detection import DetectionDelayMean


def test_detection_delay_mean_basic():
    """Test basic functionality and edge cases."""
    metric = DetectionDelayMean()

    # Single true event, early detection
    y_true = pd.DataFrame({"ilocs": [100]})
    y_pred = pd.DataFrame({"ilocs": [90]})
    assert metric(y_true, y_pred) == 0.0

    # On-time
    assert metric(y_true, pd.DataFrame({"ilocs": [100]})) == 0.0

    # Delayed by 10
    score = metric(y_true, pd.DataFrame({"ilocs": [110]}))
    assert score == 10.0

    # Multiple events
    y_true = pd.DataFrame({"ilocs": [100, 200]})
    y_pred = pd.DataFrame({"ilocs": [105, 195]})
    # First true matched with 105 → delay 5
    # Second true has no later pred → large penalty (default 1000)
    assert metric(y_true, y_pred) > 500  # rough check

    # No true events
    assert metric(pd.DataFrame({"ilocs": []}), y_pred) == 0.0

    # No predictions
    assert metric(y_true, pd.DataFrame({"ilocs": []})) > 500


def test_detection_delay_mean_params():
    """Test with non-default parameters."""
    metric = DetectionDelayMean(early_tolerance=5, max_delay=50)

    y_true = pd.DataFrame({"ilocs": [100]})
    y_pred = pd.DataFrame(
        {"ilocs": [90]}
    )  # early by 10 → clipped to 5 early → becomes 0
    assert metric(y_true, y_pred) == 0.0

    y_pred_late = pd.DataFrame({"ilocs": [200]})
    assert metric(y_true, y_pred_late) == 50.0  # capped at max_delay
