"""Tests for sliding window detector."""
import pandas as pd
import numpy as np
import pytest
from sktime.detection.sliding_window import SlidingWindowZScoreDetector

def test_zscore_detection():
    """Test basic z-score detection."""
    y = pd.DataFrame([1, 2, 3, 2, 1, 2, 3, 100, 101, 2, 3])
    d = SlidingWindowZScoreDetector(window_size=5, threshold=3.0)
    y_pred = d.fit_predict(y)
    assert 7 in y_pred['ilocs'].values


def test_quantile_detection():
    """Test quantile method."""
    y = pd.DataFrame([1, 2, 3, 2, 1, 2, 3, 100, 101, 2, 3])
    d = SlidingWindowZScoreDetector(window_size=5, method='quantile', threshold=95)
    y_pred = d.fit_predict(y)
    assert len(y_pred) > 0


def test_get_test_params():
    """Test get_test_params returns proper format."""
    params = SlidingWindowZScoreDetector.get_test_params()
    assert isinstance(params, list)
    assert len(params) >= 2


def test_no_anomalies():
    """Test when there are no anomalies."""
    y = pd.DataFrame([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])
    d = SlidingWindowZScoreDetector(window_size=5, threshold=3.0)
    y_pred = d.fit_predict(y)
    assert len(y_pred) == 0


def test_invalid_method():
    """Test invalid method raises error."""
    with pytest.raises(ValueError):
        SlidingWindowZScoreDetector(method="invalid")

        
def test_window_size_parameter():
    """Test different window sizes."""
    y = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    d = SlidingWindowZScoreDetector(window_size=3, threshold=2.0)
    y_pred = d.fit_predict(y)
    assert isinstance(y_pred, pd.DataFrame)