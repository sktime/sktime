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


@pytest.mark.skipif(
    not run_test_for_class([WindowedPrecision, WindowedRecall, WindowedF1Score]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_windowed_precision_recall_asymmetric_margins():
    """Test asymmetric backward and forward windows."""
    y_true = pd.DataFrame({"ilocs": [10]})
    y_pred = pd.DataFrame({"ilocs": [8, 11]})

    precision = WindowedPrecision(margin_backward=2, margin_forward=0)(y_true, y_pred)
    recall = WindowedRecall(margin_backward=2, margin_forward=0)(y_true, y_pred)
    f1_score = WindowedF1Score(margin_backward=2, margin_forward=0)(y_true, y_pred)

    assert precision == 0.5
    assert recall == 1.0
    assert f1_score == pytest.approx(2 / 3)


@pytest.mark.skipif(
    not run_test_for_class([WindowedPrecision, WindowedRecall, WindowedF1Score]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_windowed_precision_recall_loc_based():
    """Test loc-based windowing against a non-uniform index."""
    y_true = pd.DataFrame({"ilocs": [1]})
    y_pred = pd.DataFrame({"ilocs": [2]})
    X = pd.DataFrame(
        {"signal": [0, 1, 2, 3]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-08"]),
    )

    precision = WindowedPrecision(margin=pd.Timedelta("2D"), use_loc=True)(
        y_true, y_pred, X=X
    )
    recall = WindowedRecall(margin=pd.Timedelta("2D"), use_loc=True)(
        y_true, y_pred, X=X
    )
    f1_score = WindowedF1Score(margin=pd.Timedelta("2D"), use_loc=True)(
        y_true, y_pred, X=X
    )

    assert precision == 1.0
    assert recall == 1.0
    assert f1_score == 1.0
