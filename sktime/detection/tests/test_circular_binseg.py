# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Tests for the CircularBinarySegmentation detector."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection._change_scores._from_cost import ChangeScore
from sktime.detection.circular_binseg import CircularBinarySegmentation
from sktime.detection.costs import L2Cost
from sktime.tests.test_switch import run_test_module_changed


def _make_anomaly_data(
    n=100, p=1, anomaly_start=30, anomaly_end=50, shift=20.0, seed=42
):
    """Generate test data with a known anomaly segment."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)))
    X.iloc[anomaly_start:anomaly_end] += shift
    return X


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_cbs_invalid_score_string():
    """Test CBS rejects a string as anomaly_score."""
    with pytest.raises(ValueError, match="anomaly_score"):
        CircularBinarySegmentation("l2")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_cbs_invalid_score_change_score():
    """Test CBS rejects a ChangeScore as anomaly_score."""
    with pytest.raises(ValueError, match="anomaly_score"):
        CircularBinarySegmentation(ChangeScore(L2Cost()))


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_cbs_detects_anomaly():
    """Test CBS detects a clear segment anomaly."""
    X = _make_anomaly_data(n=100, anomaly_start=30, anomaly_end=50, shift=20.0)
    detector = CircularBinarySegmentation(penalty=20.0, min_segment_length=3)
    anomalies = detector.fit_predict(X)["ilocs"]

    assert len(anomalies) >= 1
    assert hasattr(anomalies.iloc[0], "left")
    found = any(a.left <= 40 < a.right for a in anomalies)
    assert found


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_cbs_multiple_anomalies():
    """Test CBS detects multiple non-overlapping anomalies."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((100, 1)))
    X.iloc[20:30] += 15.0
    X.iloc[60:70] += 15.0

    detector = CircularBinarySegmentation(penalty=15.0, min_segment_length=3)
    anomalies = detector.fit_predict(X)["ilocs"]

    assert len(anomalies) >= 1
    assert all(hasattr(a, "left") for a in anomalies)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_cbs_empty_result():
    """Test CBS with very high penalty produces no anomalies."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((100, 1)))
    detector = CircularBinarySegmentation(penalty=1e6)
    result = detector.fit_predict(X)
    assert "ilocs" in result.columns
    assert len(result) == 0
