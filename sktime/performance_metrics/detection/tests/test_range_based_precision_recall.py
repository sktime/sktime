"""Tests for Range-based Precision and Recall detection metrics."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection._range_based_precision_recall import (
    RangeBasedPrecision,
    RangeBasedRecall,
)
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class([RangeBasedPrecision, RangeBasedRecall]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_range_based_precision_flat():
    """Test RangeBasedPrecision with flat (default) positional bias.

    Setup
    -----
    y_true ilocs: [0, 2, 3]  -> true event range [0, 3], length 4
    y_pred ilocs: [0, 1, 3, 4, 5]

    With flat bias, each position in the range has equal weight 1/4.
    Predicted points 0 and 3 are in the true set; 1, 4, 5 are not.
    overlap contribution per hit = 1/4 (one position out of 4, equally weighted).
    Total score = (1/4 + 0 + 1/4 + 0 + 0) / 5 = 0.5 / 5 = 0.1
    """
    y_true = pd.DataFrame({"ilocs": [0, 2, 3]})
    y_pred = pd.DataFrame({"ilocs": [0, 1, 3, 4, 5]})

    metric = RangeBasedPrecision()

    loss = metric(y_true, y_pred)
    assert isinstance(loss, float)
    assert abs(loss - 0.1) < 1e-9


@pytest.mark.skipif(
    not run_test_for_class([RangeBasedPrecision, RangeBasedRecall]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_range_based_recall_flat():
    """Test RangeBasedRecall with flat (default) positional bias.

    Setup
    -----
    y_true ilocs: [0, 2, 3]  -> true event range [0, 3], length 4
    y_pred ilocs: [0, 1, 3, 4, 5]

    With flat bias, total weight = 3 (one per true point: 0, 2, 3).
    Detected true points: 0 (weight 1/4) and 3 (weight 1/4).
    Recall = (1/4 + 1/4) / (3 * 1/4) = 0.5 / 0.75 = 0.6667
    """
    y_true = pd.DataFrame({"ilocs": [0, 2, 3]})
    y_pred = pd.DataFrame({"ilocs": [0, 1, 3, 4, 5]})

    metric = RangeBasedRecall()

    loss = metric(y_true, y_pred)
    assert isinstance(loss, float)
    assert abs(loss - 2 / 3) < 1e-9


@pytest.mark.skipif(
    not run_test_for_class([RangeBasedPrecision, RangeBasedRecall]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_range_based_recall_front_bias():
    """Test RangeBasedRecall rewards earlier detections with front bias.

    Setup
    -----
    y_true ilocs: [0, 2, 3]  -> true event range [0, 3], length 4
    y_pred ilocs: [0, 1, 3, 4, 5]

    Front bias weights for length 4: pos0=1.0, pos1=0.75, pos2=0.5, pos3=0.25
    True points at positions: 0 (weight 1.0), 2 (weight 0.5), 3 (weight 0.25)
    Total weight = 1.75
    Detected: 0 (1.0/1.75) and 3 (0.25/1.75)
    Recall = (1.0 + 0.25) / 1.75 = 1.25 / 1.75 ~= 0.7143

    Front bias gives higher recall than flat (0.7143 > 0.6667) because the
    earliest detection at position 0 is rewarded more.
    """
    y_true = pd.DataFrame({"ilocs": [0, 2, 3]})
    y_pred = pd.DataFrame({"ilocs": [0, 1, 3, 4, 5]})

    metric_front = RangeBasedRecall(bias="front")
    metric_flat = RangeBasedRecall(bias="flat")

    recall_front = metric_front(y_true, y_pred)
    recall_flat = metric_flat(y_true, y_pred)

    assert isinstance(recall_front, float)
    assert abs(recall_front - 5 / 7) < 1e-9
    # front bias should reward early detection more than flat
    assert recall_front > recall_flat


@pytest.mark.skipif(
    not run_test_for_class([RangeBasedPrecision, RangeBasedRecall]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_range_based_perfect_prediction():
    """Test that perfect prediction gives Precision=0.25 and Recall=1.0 (flat bias).

    When y_pred == y_true == [0, 2, 3], range length is 4.
    Precision: each of the 3 predicted points hits a true point.
    Each contributes weight 1/4 (flat). Total = 3*(1/4) / 3 = 1/4 = 0.25.
    Recall: all 3 true points detected. Total weight = 3*(1/4) = 0.75. Score = 1.0.
    """
    y_true = pd.DataFrame({"ilocs": [0, 2, 3]})
    y_pred = pd.DataFrame({"ilocs": [0, 2, 3]})

    prec = RangeBasedPrecision()(y_true, y_pred)
    rec = RangeBasedRecall()(y_true, y_pred)

    assert isinstance(prec, float)
    assert isinstance(rec, float)
    assert abs(prec - 0.25) < 1e-9
    assert abs(rec - 1.0) < 1e-9


@pytest.mark.skipif(
    not run_test_for_class([RangeBasedPrecision, RangeBasedRecall]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_range_based_no_overlap():
    """Test that non-overlapping predictions give score 0.0."""
    y_true = pd.DataFrame({"ilocs": [0, 2, 3]})
    y_pred = pd.DataFrame({"ilocs": [10, 11, 12]})

    prec = RangeBasedPrecision()(y_true, y_pred)
    rec = RangeBasedRecall()(y_true, y_pred)

    assert prec == 0.0
    assert rec == 0.0


@pytest.mark.skipif(
    not run_test_for_class([RangeBasedPrecision, RangeBasedRecall]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_range_based_empty_inputs():
    """Test edge cases: both empty returns 1.0, one empty returns 0.0."""
    empty = pd.DataFrame({"ilocs": []})
    y = pd.DataFrame({"ilocs": [1, 2, 3]})

    # both empty -> 1.0
    assert RangeBasedPrecision()(empty, empty) == 1.0
    assert RangeBasedRecall()(empty, empty) == 1.0

    # pred empty, true not -> 0.0
    assert RangeBasedPrecision()(y, empty) == 0.0
    assert RangeBasedRecall()(y, empty) == 0.0