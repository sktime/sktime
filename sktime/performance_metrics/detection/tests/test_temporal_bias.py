"""Tests for the temporal bias metric."""

import numpy as np
from sktime.performance_metrics.detection._temporal_bias import TemporalBias

def test_temporal_bias_perfect_match():
    """Test that perfect predictions result in zero bias."""
    metric = TemporalBias()
    y_true = [10, 50, 80]
    y_pred = [10, 50, 80]

    score = metric._evaluate(y_true, y_pred)
    assert score == 0.0

def test_temporal_bias_with_delay():
    """Test that delayed predictions calculate distance correctly."""
    metric = TemporalBias()
    # Actuals at 70, 80. Predictions are late by 5 units.
    y_true = [70, 80]
    y_pred = [75, 85]

    score = metric._evaluate(y_true, y_pred)
    # The bias for each is 5. Average bias should be 5.0
    assert score == 5.0
