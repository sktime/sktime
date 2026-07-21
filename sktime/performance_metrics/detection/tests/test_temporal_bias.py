"""Tests for the Temporal Bias metric."""

import numpy as np
from sktime.performance_metrics.detection._temporal_bias import TemporalBias

def test_temporal_bias_perfect_match():
    """Test that perfect predictions result in zero bias."""
    metric = TemporalBias()
    y_true = np.array([20, 50, 80])
    y_pred = np.array([20, 50, 80])

    # If predictions are exactly on the true events, distance is 0
    score = metric._evaluate(y_true, y_pred)
    assert score == 0.0

def test_temporal_bias_with_delay():
    """Test that delayed predictions calculate distance correctly."""
    metric = TemporalBias()
    # True events at 20 and 50. Predictions are late by 5 units.
    y_true = np.array([20, 50])
    y_pred = np.array([25, 55])
    
    score = metric._evaluate(y_true, y_pred)
    # The bias distance for each is 5. Average bias should be 5.0
    assert score == 5.0

def test_temporal_bias_empty_inputs():
    """Test edge case with no predictions or true events."""
    metric = TemporalBias()
    score1 = metric._evaluate(np.array([]), np.array([10, 20]))
    score2 = metric._evaluate(np.array([10, 20]), np.array([]))
    
    assert score1 == 0.0
    assert score2 == 0.0
