"""Tests for WindowedPrecision and WindowedRecall detection metrics."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection import (
    WindowedPrecision,
    WindowedRecall,
)
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(WindowedPrecision),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
class TestWindowedPrecision:
    """Tests for WindowedPrecision metric."""

    def test_exact_match(self):
        """Test precision with exact matches, margin=0."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [2, 5, 8]})
        metric = WindowedPrecision(margin=0)
        assert metric(y_true, y_pred) == 1.0

    def test_no_match(self):
        """Test precision when no predicted event matches."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [0, 10, 20]})
        metric = WindowedPrecision(margin=0)
        assert metric(y_true, y_pred) == 0.0

    def test_partial_match(self):
        """Test precision with some matches."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [2, 10]})
        metric = WindowedPrecision(margin=0)
        assert metric(y_true, y_pred) == 0.5

    def test_margin_match(self):
        """Test precision with margin allowing near-matches."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [2, 4, 9]})
        metric = WindowedPrecision(margin=1)
        assert metric(y_true, y_pred) == 1.0

    def test_multiple_predictions_within_margin_single_gt(self):
        """Test precision when multiple predictions fall within margin of one gt."""
        y_true = pd.DataFrame({"ilocs": [10]})
        # All predictions lie within margin=1 of the single ground-truth event at 10.
        # Only one prediction should be counted as a true positive;
        # the others as false positives.
        y_pred = pd.DataFrame({"ilocs": [9, 10, 11]})
        metric = WindowedPrecision(margin=1)
        assert metric(y_true, y_pred) == pytest.approx(1 / 3)

    def test_boundary_at_margin(self):
        """Test that abs(diff) == margin counts as a match."""
        y_true = pd.DataFrame({"ilocs": [10]})
        y_pred = pd.DataFrame({"ilocs": [15]})
        metric = WindowedPrecision(margin=5)
        assert metric(y_true, y_pred) == 1.0

    def test_boundary_beyond_margin(self):
        """Test that abs(diff) > margin does not count as a match."""
        y_true = pd.DataFrame({"ilocs": [10]})
        y_pred = pd.DataFrame({"ilocs": [16]})
        metric = WindowedPrecision(margin=5)
        assert metric(y_true, y_pred) == 0.0

    def test_both_empty(self):
        """Test precision when both y_true and y_pred are empty."""
        y_true = pd.DataFrame({"ilocs": []})
        y_pred = pd.DataFrame({"ilocs": []})
        metric = WindowedPrecision(margin=0)
        assert metric(y_true, y_pred) == 1.0

    def test_empty_pred(self):
        """Test precision when predictions are empty."""
        y_true = pd.DataFrame({"ilocs": [2, 5]})
        y_pred = pd.DataFrame({"ilocs": []})
        metric = WindowedPrecision(margin=0)
        assert metric(y_true, y_pred) == 0.0

    def test_empty_gt(self):
        """Test precision when ground truth is empty but predictions exist."""
        y_true = pd.DataFrame({"ilocs": []})
        y_pred = pd.DataFrame({"ilocs": [2, 5]})
        metric = WindowedPrecision(margin=0)
        assert metric(y_true, y_pred) == 0.0


@pytest.mark.skipif(
    not run_test_for_class(WindowedRecall),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
class TestWindowedRecall:
    """Tests for WindowedRecall metric."""

    def test_exact_match(self):
        """Test recall with exact matches, margin=0."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [2, 5, 8]})
        metric = WindowedRecall(margin=0)
        assert metric(y_true, y_pred) == 1.0

    def test_no_match(self):
        """Test recall when no ground truth event is detected."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [0, 10, 20]})
        metric = WindowedRecall(margin=0)
        assert metric(y_true, y_pred) == 0.0

    def test_partial_match(self):
        """Test recall with some ground truth events detected."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [2]})
        metric = WindowedRecall(margin=0)
        result = metric(y_true, y_pred)
        assert abs(result - 1.0 / 3.0) < 1e-9

    def test_margin_match(self):
        """Test recall with margin allowing near-matches."""
        y_true = pd.DataFrame({"ilocs": [2, 5, 8]})
        y_pred = pd.DataFrame({"ilocs": [2, 4, 9]})
        metric = WindowedRecall(margin=1)
        assert metric(y_true, y_pred) == 1.0

    def test_boundary_at_margin(self):
        """Test that abs(diff) == margin counts as a match."""
        y_true = pd.DataFrame({"ilocs": [10]})
        y_pred = pd.DataFrame({"ilocs": [15]})
        metric = WindowedRecall(margin=5)
        assert metric(y_true, y_pred) == 1.0

    def test_boundary_beyond_margin(self):
        """Test that abs(diff) > margin does not count as a match."""
        y_true = pd.DataFrame({"ilocs": [10]})
        y_pred = pd.DataFrame({"ilocs": [16]})
        metric = WindowedRecall(margin=5)
        assert metric(y_true, y_pred) == 0.0

    def test_both_empty(self):
        """Test recall when both y_true and y_pred are empty."""
        y_true = pd.DataFrame({"ilocs": []})
        y_pred = pd.DataFrame({"ilocs": []})
        metric = WindowedRecall(margin=0)
        assert metric(y_true, y_pred) == 1.0

    def test_empty_pred(self):
        """Test recall when predictions are empty but ground truth exists."""
        y_true = pd.DataFrame({"ilocs": [2, 5]})
        y_pred = pd.DataFrame({"ilocs": []})
        metric = WindowedRecall(margin=0)
        assert metric(y_true, y_pred) == 0.0

    def test_empty_gt(self):
        """Test recall when ground truth is empty."""
        y_true = pd.DataFrame({"ilocs": []})
        y_pred = pd.DataFrame({"ilocs": [2, 5]})
        metric = WindowedRecall(margin=0)
        assert metric(y_true, y_pred) == 0.0

    def test_multiple_gt_within_margin_single_prediction(self):
        """Ensure one prediction cannot match multiple ground-truth events."""
        y_true = pd.DataFrame({"ilocs": [10, 11]})
        y_pred = pd.DataFrame({"ilocs": [10]})

        metric = WindowedRecall(margin=1)

        # only one GT should be matched
        assert metric(y_true, y_pred) == pytest.approx(1 / 2)
