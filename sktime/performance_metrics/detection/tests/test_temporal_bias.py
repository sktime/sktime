"""Tests for TemporalBias metric."""

import numpy as np
import pandas as pd
import pytest
from sktime.performance_metrics.detection._temporal_bias import TemporalBias
from sktime.tests.test_switch import run_test_for_class

@pytest.mark.skipif(
    not run_test_for_class(TemporalBias),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_temporal_bias_normal_case():
    """Test the temporal bias metric for the normal case,
    which is when all the predicted events occur before and after all the real events."""
    y_true = pd.DataFrame({'ilocs': [4, 5, 6, 11, 12]})
    y_pred = pd.DataFrame({'ilocs': [1, 2, 3, 6, 7, 10, 14]})
    
    metric = TemporalBias()
    skewness = metric.evaluate(y_true, y_pred)
    assert skewness == pytest.approx(0.27154541788363973)
    assert isinstance(metric.temporal_bias, list)
    assert len(metric.temporal_bias) == len(y_true)
    assert metric.temporal_bias == [-1, 1, 0, -1, -2]

@pytest.mark.skipif(
    not run_test_for_class(TemporalBias),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_temporal_bias_edge_case():
    """Test the temporal bias metric for the edge case
    when the predicted events occur only after the first real event and before the last real event."""
    y_true = pd.DataFrame({'ilocs': [4, 5, 6, 11, 12]})
    y_pred = pd.DataFrame({'ilocs': [5, 6, 8, 10, 11]})
    metric = TemporalBias()
    skewness = metric.evaluate(y_true, y_pred)
    assert skewness == pytest.approx(0.0)
    assert isinstance(metric.temporal_bias, list)
    assert len(metric.temporal_bias) == len(y_true)
    assert metric.temporal_bias == [1, 0, 0, 0, -1]
    
@pytest.mark.skipif(
    not run_test_for_class(TemporalBias),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_reject_bad_inputs():
    """Test if ValueError is thrown if y_pred is an empty array."""
    y_true = pd.DataFrame({'ilocs': [4, 5, 6, 11, 12]})
    y_pred = pd.DataFrame({'ilocs': []})
    metric = TemporalBias()
    with pytest.raise(ValueError):
        skewness = metric.evaluate(y_true, y_pred)
    
