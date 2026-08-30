"""Tests for DetectionDelayMean metric."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection._detectiondelaymean import DetectionDelayMean


def test_detection_delay_mean_basic():
    """Test basic functionality and edge cases."""
    metric = DetectionDelayMean()

    # Single true event
    y_true = pd.DataFrame({"ilocs": [100]})
    assert metric(y_true, pd.DataFrame({"ilocs": [100]})) == 0.0
    assert metric(y_true, pd.DataFrame({"ilocs": [110]})) == 10.0

    # Multiple events - only 1 prediction so second event is unmatched → penalty
    y_true = pd.DataFrame({"ilocs": [100, 200]})
    y_pred = pd.DataFrame({"ilocs": [105]})
    score = metric(y_true, y_pred)
    assert score > 500  # second event unmatched → large penalty

    # No true events → always 0
    assert metric(pd.DataFrame({"ilocs": []}), y_pred) == 0.0

    # No predictions → all events unmatched → large penalty
    assert metric(y_true, pd.DataFrame({"ilocs": []})) > 500


def test_detection_delay_mean_early_tolerance():
    """Test early_tolerance parameter."""
    metric = DetectionDelayMean(early_tolerance=5)

    y_true = pd.DataFrame({"ilocs": [100]})
    # Within tolerance (100 - 5 = 95), so delay = 0
    assert metric(y_true, pd.DataFrame({"ilocs": [97]})) == 0.0
    # Too early (before 95), so unmatched → large penalty
    assert metric(y_true, pd.DataFrame({"ilocs": [94]})) > 500


def test_detection_delay_mean_max_delay():
    """Test max_delay as cap and penalty."""
    metric = DetectionDelayMean(max_delay=50)

    y_true = pd.DataFrame({"ilocs": [100]})
    # Very late prediction → capped at max_delay
    assert metric(y_true, pd.DataFrame({"ilocs": [200]})) == 50.0
    # No prediction → penalty = max_delay
    assert metric(y_true, pd.DataFrame({"ilocs": []})) == 50.0


def test_detection_delay_mean_combined_params():
    """Test combination of parameters."""
    metric = DetectionDelayMean(early_tolerance=5, max_delay=30)

    y_true = pd.DataFrame({"ilocs": [100, 200]})
    y_pred = pd.DataFrame({"ilocs": [96, 250]})
    # 96 >= 100-5=95 → delay = max(0, 96-100) = 0
    # 250 >= 200-5=195 → delay = min(250-200, 30) = 30
    # mean = (0 + 30) / 2 = 15.0
    score = metric(y_true, y_pred)
    assert score == 15.0


def test_detection_delay_mean_get_test_params():
    """Test get_test_params returns valid parameter sets."""
    params = DetectionDelayMean.get_test_params()
    assert len(params) >= 1
    for p in params:
        metric = DetectionDelayMean(**p)
        assert isinstance(metric, DetectionDelayMean)


@pytest.mark.parametrize(
    "y_true_ilocs, y_pred_ilocs, expected",
    [
        ([100], [110], 10.0),  # late by 10
        ([100], [100], 0.0),  # exact match → 0
        ([], [100], 0.0),  # no true events → 0
        ([100, 200], [105, 210], 7.5),  # delays 5 and 10, mean=7.5
    ],
)
def test_detection_delay_mean_parametrized(y_true_ilocs, y_pred_ilocs, expected):
    """Parametrized tests."""
    metric = DetectionDelayMean()
    y_true = pd.DataFrame({"ilocs": y_true_ilocs})
    y_pred = pd.DataFrame({"ilocs": y_pred_ilocs})
    result = metric(y_true, y_pred)
    assert abs(result - expected) < 1e-6
