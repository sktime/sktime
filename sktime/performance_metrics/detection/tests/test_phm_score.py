"""Tests for the PHM 2008 asymmetric detection score."""

import math

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.detection._phm_score import PHMScore
from sktime.tests.test_switch import run_test_for_class

# All tests below share the same skip guard, so define it once.
pytestmark = pytest.mark.skipif(
    not run_test_for_class(PHMScore),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def _df(ilocs):
    """Shorthand for building a points-style detection DataFrame."""
    return pd.DataFrame({"ilocs": list(ilocs)})


def test_perfect_match_returns_zero():
    """Every prediction exactly at its true event → score is exactly 0."""
    y_true = _df([100, 200, 300])
    y_pred = _df([100, 200, 300])
    assert PHMScore()(y_true, y_pred) == 0.0


def test_one_step_early_matches_phm_formula():
    """Single early delay d=-1 should match exp(1/13) - 1 exactly."""
    y_true = _df([100])
    y_pred = _df([99])
    metric = PHMScore(unmatched_penalty=1000.0)
    expected = math.exp(1 / 13) - 1
    assert math.isclose(metric(y_true, y_pred), expected, abs_tol=1e-12)


def test_one_step_late_matches_phm_formula():
    """Single late delay d=+1 should match exp(1/10) - 1 exactly."""
    y_true = _df([100])
    y_pred = _df([101])
    metric = PHMScore(unmatched_penalty=1000.0)
    expected = math.exp(1 / 10) - 1
    assert math.isclose(metric(y_true, y_pred), expected, abs_tol=1e-12)


def test_late_is_strictly_worse_than_early_for_same_magnitude():
    """Asymmetry: for the same |d|, late should cost more than early."""
    y_true = _df([100])
    metric = PHMScore(unmatched_penalty=1000.0)
    for k in [1, 5, 10, 20]:
        early = metric(y_true, _df([100 - k]))
        late = metric(y_true, _df([100 + k]))
        assert late > early, f"late ({late}) not > early ({early}) for k={k}"


def test_greedy_matching_multiple_events():
    """Multiple events: greedy match by nearest, sum of PHM costs."""
    y_true = _df([100, 200, 300])
    y_pred = _df([98, 205, 290])
    metric = PHMScore(unmatched_penalty=1000.0)
    expected = (
        (math.exp(2 / 13) - 1)  # d = 98 - 100 = -2
        + (math.exp(5 / 10) - 1)  # d = 205 - 200 = +5
        + (math.exp(10 / 13) - 1)  # d = 290 - 300 = -10
    )
    assert math.isclose(metric(y_true, y_pred), expected, abs_tol=1e-12)


def test_missed_detection_returns_inf_by_default():
    """Missing a true event with the default penalty produces inf."""
    y_true = _df([100, 200])
    y_pred = _df([100])
    assert PHMScore()(y_true, y_pred) == math.inf


def test_false_alarm_returns_inf_by_default():
    """A spurious prediction with the default penalty produces inf."""
    y_true = _df([100])
    y_pred = _df([100, 500])
    assert PHMScore()(y_true, y_pred) == math.inf


def test_missed_detection_with_finite_penalty():
    """With a finite penalty, one missed event adds exactly that penalty."""
    y_true = _df([100, 200])
    y_pred = _df([100])
    assert PHMScore(unmatched_penalty=50.0)(y_true, y_pred) == 50.0


def test_false_alarm_with_finite_penalty():
    """With a finite penalty, one false alarm adds exactly that penalty."""
    y_true = _df([100])
    y_pred = _df([100, 500])
    assert PHMScore(unmatched_penalty=50.0)(y_true, y_pred) == 50.0


def test_both_empty_returns_zero():
    """No true events and no predictions → no work to do, score is 0."""
    empty = _df([])
    assert PHMScore()(empty, empty) == 0.0


def test_empty_true_all_predictions_are_false_alarms():
    """All predictions are false alarms when y_true is empty."""
    metric = PHMScore(unmatched_penalty=7.0)
    assert metric(_df([]), _df([10, 20, 30])) == 21.0


def test_empty_pred_all_true_are_missed():
    """All true events are misses when y_pred is empty."""
    metric = PHMScore(unmatched_penalty=7.0)
    assert metric(_df([10, 20, 30]), _df([])) == 21.0


def test_mean_aggregation_normalizes_by_event_count():
    """aggregation='mean' divides the sum by the number of events."""
    y_true = _df([100, 200])
    y_pred = _df([100, 202])
    s_sum = PHMScore(aggregation="sum")(y_true, y_pred)
    s_mean = PHMScore(aggregation="mean")(y_true, y_pred)
    assert math.isclose(s_mean * 2, s_sum, abs_tol=1e-12)


def test_tau_parameters_control_penalty_steepness():
    """Smaller tau → steeper penalty for the same delay."""
    y_true = _df([100])
    y_pred = _df([105])
    strict = PHMScore(tau_late=5.0, unmatched_penalty=1000.0)(y_true, y_pred)
    lenient = PHMScore(tau_late=20.0, unmatched_penalty=1000.0)(y_true, y_pred)
    assert strict > lenient


def test_sum_equals_mean_when_one_event():
    """With exactly one event, sum and mean aggregations coincide."""
    y_true = _df([100])
    y_pred = _df([105])
    s_sum = PHMScore(aggregation="sum", unmatched_penalty=1000.0)(y_true, y_pred)
    s_mean = PHMScore(aggregation="mean", unmatched_penalty=1000.0)(y_true, y_pred)
    assert math.isclose(s_sum, s_mean, abs_tol=1e-12)


def test_result_is_python_float():
    """The metric must return a plain Python float, not np.float64."""
    y_true = _df([100])
    y_pred = _df([103])
    out = PHMScore(unmatched_penalty=1000.0)(y_true, y_pred)
    assert isinstance(out, float)
    assert not isinstance(out, np.floating)


def test_get_test_params_instances_all_evaluate():
    """Every parameter set from get_test_params should construct and run."""
    y_true = _df([10, 20])
    y_pred = _df([11, 19])
    for params in PHMScore.get_test_params():
        metric = PHMScore(**params)
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert result >= 0.0
