"""Tests for WindowedF1Score detection metric."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection._f1score import WindowedF1Score
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(WindowedF1Score),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
class TestWindowedF1Score:
    """Tests for WindowedF1Score metric."""

    def test_exact_match(self):
        """Test F1 with exact matches, margin=0."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [2, 5, 8]})
        metric = WindowedF1Score(margin=0)
        assert metric(y_true, y_pred) == 1.0

    def test_no_match(self):
        """Test F1 when no predicted event matches any ground-truth event."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [0, 10, 20]})
        metric = WindowedF1Score(margin=0)
        assert metric(y_true, y_pred) == 0.0

    def test_both_empty(self):
        """Test F1 when both y_true and y_pred are empty."""
        y_true = pd.DataFrame({"ilocs": []})
        y_pred = pd.DataFrame({"ilocs": []})
        metric = WindowedF1Score(margin=0)
        assert metric(y_true, y_pred) == 1.0

    def test_empty_pred(self):
        """Test F1 when predictions are empty but ground truth exists."""
        y_true = pd.DataFrame({"ilocs": [2, 5]})
        y_pred = pd.DataFrame({"ilocs": []})
        metric = WindowedF1Score(margin=0)
        assert metric(y_true, y_pred) == 0.0

    def test_empty_gt(self):
        """Test F1 when ground truth is empty but predictions exist."""
        y_true = pd.DataFrame({"ilocs": []})
        y_pred = pd.DataFrame({"ilocs": [2, 5]})
        metric = WindowedF1Score(margin=0)
        assert metric(y_true, y_pred) == 0.0

    def test_boundary_at_margin(self):
        """Test that abs(diff) == margin counts as a match."""
        y_true = pd.DataFrame({"ilocs": [10]})
        y_pred = pd.DataFrame({"ilocs": [15]})
        metric = WindowedF1Score(margin=5)
        assert metric(y_true, y_pred) == 1.0

        y_pred_left = pd.DataFrame({"ilocs": [5]})
        assert metric(y_true, y_pred_left) == 1.0

    def test_boundary_beyond_margin(self):
        """Test that abs(diff) > margin does not count as a match."""
        y_true = pd.DataFrame({"ilocs": [10]})
        y_pred = pd.DataFrame({"ilocs": [16]})
        metric = WindowedF1Score(margin=5)
        assert metric(y_true, y_pred) == 0.0

    def test_multiple_predictions_within_margin_single_gt(self):
        """One ground-truth event should not match multiple predicted events.

        When multiple predicted events fall within the margin of a single
        ground-truth event, only one match is counted (one-to-one constraint).
        precision = 1/3, recall = 1/1 = 1.0, F1 = 2*(1/3)*1 / (1/3+1) = 1/2.
        """
        y_true = pd.DataFrame({"ilocs": [10]})
        y_pred = pd.DataFrame({"ilocs": [9, 10, 11]})
        metric = WindowedF1Score(margin=1)
        precision = 1 / 3
        recall = 1.0
        expected_f1 = 2 * precision * recall / (precision + recall)
        assert metric(y_true, y_pred) == pytest.approx(expected_f1)

    def test_multiple_gt_within_margin_single_prediction(self):
        """One predicted event should not match multiple ground-truth events.

        When multiple ground-truth events fall within the margin of a single
        prediction, only one match is counted (one-to-one constraint).
        precision = 1/1 = 1.0, recall = 1/2, F1 = 2*1*(1/2) / (1+1/2) = 2/3.
        """
        y_true = pd.DataFrame({"ilocs": [9, 11]})
        y_pred = pd.DataFrame({"ilocs": [10]})
        metric = WindowedF1Score(margin=1)
        precision = 1.0
        recall = 1 / 2
        expected_f1 = 2 * precision * recall / (precision + recall)
        assert metric(y_true, y_pred) == pytest.approx(expected_f1)
