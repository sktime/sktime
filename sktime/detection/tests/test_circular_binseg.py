"""Tests for CircularBinarySegmentation change point detection method."""

__author__ = ["CloseChoice"]
__all__ = []
import numpy as np
import pandas as pd
import pytest
from skchange.costs import L1Cost
from skchange.datasets import generate_anomalous_data

from sktime.detection.skchange_aseg import CircularBinarySegmentation
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(CircularBinarySegmentation),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_circular_binseg_with_params():
    """Test that the interface to skchange works correctly for all parameters."""
    detector = CircularBinarySegmentation(
        anomaly_score=L1Cost(1),
        penalty=1.0,
        min_segment_length=5,
        max_interval_length=1000,
        growth_factor=1.5,
    )
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    y = pd.Series(np.zeros(len(x)))
    fit_detector = detector.fit(x, y)
    result = fit_detector.predict(x)
    assert isinstance(detector, CircularBinarySegmentation)
    assert isinstance(fit_detector, CircularBinarySegmentation)
    assert isinstance(result, pd.DataFrame)
