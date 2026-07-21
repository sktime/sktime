"""Tests for the advance detection score."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection._advance import AdvanceDetectionScore
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(AdvanceDetectionScore),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_advance_detection_perfect():
    """Test perfect advance detection (detection exactly at event time)."""
    y_true = pd.DataFrame({"ilocs": [5, 10]})
    y_pred = pd.DataFrame({"ilocs": [5, 10]})

    metric = AdvanceDetectionScore(window=10, normalize=True)
    score = metric(y_true, y_pred)

    assert isinstance(score, float)
    # delay = 0 for both events, score = 1.0 each, mean = 1.0
    assert score == 1.0


@pytest.mark.skipif(
    not run_test_for_class(AdvanceDetectionScore),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_advance_detection_early():
    """Test early detection within window scores partial credit."""
    y_true = pd.DataFrame({"ilocs": [10, 20]})
    y_pred = pd.DataFrame({"ilocs": [5, 15]})

    metric = AdvanceDetectionScore(window=10, normalize=True)
    score = metric(y_true, y_pred)

    assert isinstance(score, float)
    # event at 10: closest advance detection at 5, delay=5, score = 1 - 5/10 = 0.5
    # event at 20: closest advance detection at 15, delay=5, score = 1 - 5/10 = 0.5
    # mean = 0.5
    assert score == 0.5


@pytest.mark.skipif(
    not run_test_for_class(AdvanceDetectionScore),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_advance_detection_late():
    """Test late detection (after the event) scores 0 by default."""
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [8]})

    metric = AdvanceDetectionScore(window=10, normalize=True)
    score = metric(y_true, y_pred)

    assert isinstance(score, float)
    # detection at 8 is after event at 5, all detections are late, score = 0
    assert score == 0.0


@pytest.mark.skipif(
    not run_test_for_class(AdvanceDetectionScore),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_advance_detection_late_penalty():
    """Test late detection with nonzero penalty."""
    y_true = pd.DataFrame({"ilocs": [5]})
    y_pred = pd.DataFrame({"ilocs": [8]})

    metric = AdvanceDetectionScore(window=10, penalty_late=0.5, normalize=True)
    score = metric(y_true, y_pred)

    assert isinstance(score, float)
    assert score == 0.5


@pytest.mark.skipif(
    not run_test_for_class(AdvanceDetectionScore),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_advance_detection_outside_window():
    """Test detection outside the window scores 0."""
    y_true = pd.DataFrame({"ilocs": [20]})
    y_pred = pd.DataFrame({"ilocs": [5]})

    metric = AdvanceDetectionScore(window=10, normalize=True)
    score = metric(y_true, y_pred)

    assert isinstance(score, float)
    # detection at 5 is 15 time units before event at 20, outside window of 10
    assert score == 0.0


@pytest.mark.skipif(
    not run_test_for_class(AdvanceDetectionScore),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_advance_detection_unnormalized():
    """Test unnormalized score returns sum instead of mean."""
    y_true = pd.DataFrame({"ilocs": [10, 20]})
    y_pred = pd.DataFrame({"ilocs": [5, 15]})

    metric = AdvanceDetectionScore(window=10, normalize=False)
    score = metric(y_true, y_pred)

    assert isinstance(score, float)
    # event at 10: score = 0.5, event at 20: score = 0.5
    # sum = 1.0
    assert score == 1.0


@pytest.mark.skipif(
    not run_test_for_class(AdvanceDetectionScore),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_advance_detection_with_X():
    """Test advance detection with custom X index."""
    y_true = pd.DataFrame({"ilocs": [2, 4]})  # locs are 20, 40
    y_pred = pd.DataFrame({"ilocs": [1, 3]})  # locs are 10, 30
    X = pd.DataFrame(
        {"val": [0, 1, 2, 3, 4, 5]}, index=[0, 10, 20, 30, 40, 50]
    )

    metric = AdvanceDetectionScore(window=15, normalize=True)
    score = metric(y_true, y_pred, X)

    assert isinstance(score, float)
    # event at loc=20: closest advance detection at loc=10, delay=10, score=1-10/15
    # event at loc=40: closest advance detection at loc=30, delay=10, score=1-10/15
    # mean = 1 - 10/15 = 1/3
    assert abs(score - (1.0 - 10.0 / 15.0)) < 1e-10


@pytest.mark.skipif(
    not run_test_for_class(AdvanceDetectionScore),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_advance_detection_mixed():
    """Test mix of advance, late, and out-of-window detections."""
    y_true = pd.DataFrame({"ilocs": [10, 20, 30]})
    y_pred = pd.DataFrame({"ilocs": [8, 25]})

    metric = AdvanceDetectionScore(window=5, normalize=True)
    score = metric(y_true, y_pred)

    assert isinstance(score, float)
    # event at 10: advance detection at 8, delay=2, score = 1 - 2/5 = 0.6
    # event at 20: advance detection at 8, delay=12, outside window -> 0
    #              detection at 25 is late -> check advance only: 8 is advance but
    #              delay=12 > window=5, so score = 0
    # event at 30: advance detection at 25, delay=5, score = 1 - 5/5 = 0.0
    #              also detection at 8, delay=22, outside window
    # mean = (0.6 + 0.0 + 0.0) / 3 = 0.2
    assert abs(score - 0.2) < 1e-10
