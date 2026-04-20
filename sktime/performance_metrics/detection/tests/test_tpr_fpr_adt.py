# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for WindowedTPR, WindowedFPR, and EarlyDetectionTime metrics."""

__author__ = ["rupeshca007"]

import math

import pandas as pd
import pytest

from sktime.performance_metrics.detection._tpr_fpr_adt import (
    EarlyDetectionTime,
    WindowedFPR,
    WindowedTPR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pts(ilocs):
    """Create a minimal 'points' DataFrame from a list of iloc indices."""
    return pd.DataFrame({"ilocs": ilocs})


# ===========================================================================
# WindowedTPR tests
# ===========================================================================


class TestWindowedTPR:
    """Tests for WindowedTPR."""

    def test_perfect_match_margin0(self):
        """Exact matches → TPR == 1.0."""
        y_true = _pts([10, 50, 90])
        y_pred = _pts([10, 50, 90])
        assert WindowedTPR(margin=0)(y_true, y_pred) == 1.0

    def test_no_match_margin0(self):
        """Off-by-one with margin=0 → TPR == 0.0."""
        y_true = _pts([10, 50, 90])
        y_pred = _pts([11, 51, 91])
        assert WindowedTPR(margin=0)(y_true, y_pred) == 0.0

    def test_all_match_within_margin(self):
        """All events detected within margin → TPR == 1.0."""
        y_true = _pts([10, 50, 90])
        y_pred = _pts([11, 52, 95])
        assert WindowedTPR(margin=5)(y_true, y_pred) == 1.0

    def test_partial_match(self):
        """Only 2 of 3 events detected → TPR == 2/3."""
        y_true = _pts([10, 50, 90])
        y_pred = _pts([10, 50])  # missed 90
        result = WindowedTPR(margin=0)(y_true, y_pred)
        assert abs(result - 2 / 3) < 1e-9

    def test_empty_both(self):
        """Both empty → TPR == 1.0 (convention: nothing to detect)."""
        assert WindowedTPR()(y_true=_pts([]), y_pred=_pts([])) == 1.0

    def test_empty_pred_nonempty_true(self):
        """No predictions, events exist → TPR == 0.0."""
        assert WindowedTPR()(y_true=_pts([10, 20]), y_pred=_pts([])) == 0.0

    def test_empty_true_nonempty_pred(self):
        """No true events, predictions exist → TPR == 1.0 (nothing to miss)."""
        assert WindowedTPR()(y_true=_pts([]), y_pred=_pts([5, 15])) == 1.0

    def test_no_double_counting(self):
        """A single prediction should not satisfy two ground-truth events."""
        y_true = _pts([10, 12])
        y_pred = _pts([11])  # within margin=2 of both, but can only match once
        result = WindowedTPR(margin=2)(y_true, y_pred)
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_returns_float(self):
        """Return type must be float."""
        result = WindowedTPR(margin=1)(y_true=_pts([5]), y_pred=_pts([5]))
        assert isinstance(result, float)

    @pytest.mark.parametrize("margin", [0, 1, 5, 100])
    def test_perfect_score_various_margins(self, margin):
        """Exact predictions always give TPR == 1.0 regardless of margin."""
        y_true = _pts([20, 40, 60])
        y_pred = _pts([20, 40, 60])
        assert WindowedTPR(margin=margin)(y_true, y_pred) == 1.0

    def test_get_test_params(self):
        """get_test_params returns valid list of dicts."""
        params = WindowedTPR.get_test_params()
        assert isinstance(params, list)
        for p in params:
            assert isinstance(p, dict)
            inst = WindowedTPR(**p)
            assert inst is not None


# ===========================================================================
# WindowedFPR tests
# ===========================================================================


class TestWindowedFPR:
    """Tests for WindowedFPR."""

    def test_perfect_match_no_false_alarms(self):
        """All predictions match → FPR == 0.0."""
        y_true = _pts([10, 50, 90])
        y_pred = _pts([10, 50, 90])
        assert WindowedFPR(margin=0)(y_true, y_pred) == 0.0

    def test_all_false_alarms(self):
        """No prediction matches any true event → FPR == 1.0."""
        y_true = _pts([10, 50, 90])
        y_pred = _pts([25, 70])
        assert WindowedFPR(margin=0)(y_true, y_pred) == 1.0

    def test_partial_false_alarms(self):
        """One false alarm among three predictions → FPR == 1/3."""
        y_true = _pts([10, 50, 90])
        y_pred = _pts([11, 30, 52])   # 30 is a false alarm, margin=5
        result = WindowedFPR(margin=5)(y_true, y_pred)
        assert abs(result - 1 / 3) < 1e-9

    def test_empty_pred(self):
        """No predictions → FPR == 0.0 (no false alarms possible)."""
        assert WindowedFPR()(y_true=_pts([10, 20]), y_pred=_pts([])) == 0.0

    def test_empty_true_nonempty_pred(self):
        """No true events, but predictions exist → all are false alarms → FPR == 1.0."""
        assert WindowedFPR()(y_true=_pts([]), y_pred=_pts([5, 15])) == 1.0

    def test_empty_both(self):
        """Both empty → FPR == 0.0."""
        assert WindowedFPR()(y_true=_pts([]), y_pred=_pts([])) == 0.0

    def test_returns_float(self):
        result = WindowedFPR()(y_true=_pts([10]), y_pred=_pts([10]))
        assert isinstance(result, float)

    def test_get_test_params(self):
        params = WindowedFPR.get_test_params()
        assert isinstance(params, list)
        for p in params:
            inst = WindowedFPR(**p)
            assert inst is not None


# ===========================================================================
# EarlyDetectionTime tests
# ===========================================================================


class TestEarlyDetectionTime:
    """Tests for EarlyDetectionTime."""

    def test_exact_match_zero_advance(self):
        """Exact predictions → advance time == 0.0."""
        y_true = _pts([20, 60, 100])
        y_pred = _pts([20, 60, 100])
        result = EarlyDetectionTime(margin=0)(y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_early_detection_positive(self):
        """Detector fires 5 steps early → advance time == 5.0."""
        y_true = _pts([20])
        y_pred = _pts([15])
        result = EarlyDetectionTime(margin=10)(y_true, y_pred)
        assert result == pytest.approx(5.0, abs=1e-9)

    def test_late_detection_negative(self):
        """Detector fires 3 steps late → advance time == -3.0."""
        y_true = _pts([20])
        y_pred = _pts([23])
        result = EarlyDetectionTime(margin=5)(y_true, y_pred)
        assert result == pytest.approx(-3.0, abs=1e-9)

    def test_mean_advance_mixed(self):
        """Two early (5, 5) and one late (-2) → mean == (5+5-2)/3 ≈ 2.67."""
        y_true = _pts([20, 60, 100])
        y_pred = _pts([15, 55, 102])  # advances: +5, +5, -2
        result = EarlyDetectionTime(margin=10, aggregate="mean")(y_true, y_pred)
        assert result == pytest.approx((5 + 5 - 2) / 3, abs=1e-9)

    def test_median_aggregate(self):
        """Median of [5, 5, -2] == 5.0."""
        y_true = _pts([20, 60, 100])
        y_pred = _pts([15, 55, 102])
        result = EarlyDetectionTime(margin=10, aggregate="median")(y_true, y_pred)
        assert result == pytest.approx(5.0, abs=1e-9)

    def test_min_aggregate(self):
        """Min of [5, 5, -2] == -2.0."""
        y_true = _pts([20, 60, 100])
        y_pred = _pts([15, 55, 102])
        result = EarlyDetectionTime(margin=10, aggregate="min")(y_true, y_pred)
        assert result == pytest.approx(-2.0, abs=1e-9)

    def test_max_aggregate(self):
        """Max of [5, 5, -2] == 5.0."""
        y_true = _pts([20, 60, 100])
        y_pred = _pts([15, 55, 102])
        result = EarlyDetectionTime(margin=10, aggregate="max")(y_true, y_pred)
        assert result == pytest.approx(5.0, abs=1e-9)

    def test_missed_event_excluded_by_default(self):
        """Missed event (no penalty) is excluded from the aggregate."""
        y_true = _pts([10, 50])
        y_pred = _pts([8])  # only matches 10, prediction for 50 is missing
        result = EarlyDetectionTime(margin=5)(y_true, y_pred)
        # only advance for event at 10: 10-8=2
        assert result == pytest.approx(2.0, abs=1e-9)

    def test_missed_event_with_penalty(self):
        """Missed event applies penalty value to aggregate."""
        y_true = _pts([10, 50])
        y_pred = _pts([8])  # matches event at 10 (advance=2), misses 50
        result = EarlyDetectionTime(margin=5, missed_penalty=-10)(y_true, y_pred)
        # advances: [2, -10] → mean = -4.0
        assert result == pytest.approx(-4.0, abs=1e-9)

    def test_empty_true_returns_nan(self):
        """No true events → nan (undefined)."""
        result = EarlyDetectionTime(margin=5)(y_true=_pts([]), y_pred=_pts([5]))
        assert math.isnan(result)

    def test_empty_both_returns_nan(self):
        """Both empty → nan."""
        result = EarlyDetectionTime()(y_true=_pts([]), y_pred=_pts([]))
        assert math.isnan(result)

    def test_invalid_aggregate_raises(self):
        """Invalid aggregate string → ValueError."""
        metric = EarlyDetectionTime(margin=5, aggregate="geometric_mean")
        with pytest.raises(ValueError, match="aggregate must be one of"):
            metric(y_true=_pts([10]), y_pred=_pts([10]))

    def test_returns_float(self):
        result = EarlyDetectionTime(margin=2)(y_true=_pts([10]), y_pred=_pts([8]))
        assert isinstance(result, float)

    def test_get_test_params(self):
        params = EarlyDetectionTime.get_test_params()
        assert isinstance(params, list)
        for p in params:
            inst = EarlyDetectionTime(**p)
            assert inst is not None


# ===========================================================================
# Integration: TPR + FPR + ADT together (event detection workflow)
# ===========================================================================


class TestMetricIntegration:
    """Integration tests simulating a realistic event-detection evaluation."""

    def test_good_detector(self):
        """A near-perfect detector should have high TPR, low FPR, positive ADT."""
        # 5 true events; detector fires slightly early for all 5
        y_true = _pts([100, 200, 300, 400, 500])
        y_pred = _pts([95, 195, 295, 398, 498])   # all 2-5 steps early

        tpr = WindowedTPR(margin=10)(y_true, y_pred)
        fpr = WindowedFPR(margin=10)(y_true, y_pred)
        adt = EarlyDetectionTime(margin=10)(y_true, y_pred)

        assert tpr == 1.0, f"Expected TPR=1.0, got {tpr}"
        assert fpr == 0.0, f"Expected FPR=0.0, got {fpr}"
        assert adt > 0.0, f"Expected positive ADT (early detection), got {adt}"

    def test_bad_detector_many_false_alarms(self):
        """A detector with many false alarms should have high FPR."""
        y_true = _pts([50])
        y_pred = _pts([10, 20, 30, 40, 50])  # 4 false alarms

        fpr = WindowedFPR(margin=0)(y_true, y_pred)
        assert fpr == pytest.approx(4 / 5, abs=1e-9)

    def test_tpr_fpr_complementary(self):
        """On fixed data, increasing margin should weakly increase TPR."""
        y_true = _pts([10, 30, 50])
        y_pred = _pts([13, 33, 54])  # all off by 3-4

        tpr_strict = WindowedTPR(margin=2)(y_true, y_pred)
        tpr_loose = WindowedTPR(margin=5)(y_true, y_pred)

        assert tpr_loose >= tpr_strict
