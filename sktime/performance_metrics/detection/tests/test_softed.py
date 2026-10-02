"""Tests for the SoftED F1 detection metric."""

import pandas as pd
import pytest

from sktime.performance_metrics.detection._softed import SoftEDF1Score
from sktime.tests.test_switch import run_test_for_class


@pytest.fixture
def true_events():
    """Three ground-truth events at ilocs 10, 50, 90."""
    return pd.DataFrame({"ilocs": [10, 50, 90]})


@pytest.mark.skipif(
    not run_test_for_class(SoftEDF1Score),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_softed_perfect_match(true_events):
    """Identical predictions to ground truth score exactly 1.0."""
    metric = SoftEDF1Score(tolerance=5, membership="linear")
    assert metric(true_events, true_events) == pytest.approx(1.0)


@pytest.mark.skipif(
    not run_test_for_class(SoftEDF1Score),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_softed_all_outside_tolerance(true_events):
    """Predictions outside tolerance contribute nothing, score 0.0."""
    y_pred = pd.DataFrame({"ilocs": [200, 300, 400]})
    metric = SoftEDF1Score(tolerance=5, membership="linear")
    assert metric(true_events, y_pred) == 0.0


@pytest.mark.skipif(
    not run_test_for_class(SoftEDF1Score),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_softed_rectangular_hard_cutoff(true_events):
    """Rectangular membership is a hard step function at tolerance."""
    y_pred_in = pd.DataFrame({"ilocs": [10, 50, 90]})
    # all 6 iloc away, tolerance=5
    y_pred_out = pd.DataFrame({"ilocs": [16, 56, 96]})
    metric = SoftEDF1Score(tolerance=5, membership="rectangular")
    assert metric(true_events, y_pred_in) == pytest.approx(1.0)
    assert metric(true_events, y_pred_out) == 0.0


@pytest.mark.skipif(
    not run_test_for_class(SoftEDF1Score),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_softed_linear_decay(true_events):
    """Linear membership at distance 2 with tolerance 5 gives mu = 1 - 2/5 = 0.6.

    All three predictions are 2 iloc away from their matched true event,
    so soft_matches = 3 * 0.6 = 1.8, soft_precision = soft_recall = 0.6,
    and F1 = 0.6.
    """
    y_pred = pd.DataFrame({"ilocs": [12, 52, 92]})
    metric = SoftEDF1Score(tolerance=5, membership="linear")
    assert metric(true_events, y_pred) == pytest.approx(0.6)


@pytest.mark.skipif(
    not run_test_for_class(SoftEDF1Score),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_softed_empty_both_returns_one():
    """No true events and no predictions is a trivial perfect score."""
    empty = pd.DataFrame({"ilocs": []})
    metric = SoftEDF1Score()
    assert metric(empty, empty) == 1.0


@pytest.mark.skipif(
    not run_test_for_class(SoftEDF1Score),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_softed_invalid_membership_raises(true_events):
    """An unknown membership string raises ValueError."""
    metric = SoftEDF1Score(membership="quadratic")
    with pytest.raises(ValueError, match="linear.*rectangular"):
        metric(true_events, true_events)


@pytest.mark.skipif(
    not run_test_for_class(SoftEDF1Score),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_softed_negative_tolerance_raises(true_events):
    """A negative tolerance raises ValueError."""
    metric = SoftEDF1Score(tolerance=-1)
    with pytest.raises(ValueError, match="non-negative"):
        metric(true_events, true_events)
