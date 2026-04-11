"""Tests for windowed precision and recall detection metrics."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection import (
    WindowedF1Score,
    WindowedPrecision,
    WindowedRecall,
)
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class([WindowedPrecision, WindowedRecall, WindowedF1Score]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_windowed_precision_recall():
    """Test precision and recall values under windowed matching."""
    y_true = pd.DataFrame({"ilocs": [1, 4, 10]})
    y_pred = pd.DataFrame({"ilocs": [1, 5, 8, 13]})

    precision = WindowedPrecision(margin=1)(y_true, y_pred)
    recall = WindowedRecall(margin=1)(y_true, y_pred)
    f1_score = WindowedF1Score(margin=1)(y_true, y_pred)

    assert precision == 0.5
    assert recall == 2 / 3
    assert f1_score == pytest.approx(4 / 7)


@pytest.mark.skipif(
    not run_test_for_class([WindowedPrecision, WindowedRecall, WindowedF1Score]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_windowed_matching_is_one_to_one():
    """Test that one prediction cannot match multiple true events."""
    y_true = pd.DataFrame({"ilocs": [10, 11]})
    y_pred = pd.DataFrame({"ilocs": [10]})

    precision = WindowedPrecision(margin=1)(y_true, y_pred)
    recall = WindowedRecall(margin=1)(y_true, y_pred)
    f1_score = WindowedF1Score(margin=1)(y_true, y_pred)

    assert precision == 1.0
    assert recall == 0.5
    assert f1_score == pytest.approx(2 / 3)


@pytest.mark.skipif(
    not run_test_for_class([WindowedPrecision, WindowedRecall]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_windowed_precision_recall_empty_inputs():
    """Test empty-input score conventions."""
    y_empty = pd.DataFrame({"ilocs": []})
    y_nonempty = pd.DataFrame({"ilocs": [0]})

    assert WindowedPrecision()(y_empty, y_empty) == 1.0
    assert WindowedRecall()(y_empty, y_empty) == 1.0

    assert WindowedPrecision()(y_nonempty, y_empty) == 0.0
    assert WindowedRecall()(y_nonempty, y_empty) == 0.0

    assert WindowedPrecision()(y_empty, y_nonempty) == 0.0
    assert WindowedRecall()(y_empty, y_nonempty) == 0.0
