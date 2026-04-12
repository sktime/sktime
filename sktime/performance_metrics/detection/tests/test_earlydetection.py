"""Tests for EarlyDetectionScore metric."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection import EarlyDetectionScore


def test_early_detection_score():
    """Test basic cases and edge cases for EarlyDetectionScore."""
    metric = EarlyDetectionScore()

    y_true = pd.DataFrame({"ilocs": [100]})

    # Early or on-time detection → score = 1.0
    assert metric(y_true, pd.DataFrame({"ilocs": [80]})) == 1.0
    assert metric(y_true, pd.DataFrame({"ilocs": [100]})) == 1.0

    # Delayed detection (delay = 20) → 1 / (1 + 20)
    score = metric(y_true, pd.DataFrame({"ilocs": [120]}))
    assert pytest.approx(score, rel=1e-6) == 1.0 / 21.0

    # Edge cases
    assert metric(y_true, pd.DataFrame({"ilocs": []})) == 0.0
    assert metric(pd.DataFrame({"ilocs": []}), pd.DataFrame({"ilocs": [50]})) == 0.0
