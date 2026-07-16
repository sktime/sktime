# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten
"""Generic tests for all segment anomaly detectors."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection.capa import CAPA
from sktime.detection.circular_binseg import CircularBinarySegmentation
from sktime.detection.moving_window import MovingWindow
from sktime.detection.stat_threshold import StatThresholdAnomaliser
from sktime.tests.test_switch import run_test_module_changed

SEGMENT_ANOMALY_DETECTORS = [
    CAPA,
    CircularBinarySegmentation,
    StatThresholdAnomaliser,
]

# StatThresholdAnomaliser can produce overlapping segments that
# sparse_to_dense does not support, so exclude from transform tests.
TRANSFORM_TESTABLE_DETECTORS = [
    CAPA,
    CircularBinarySegmentation,
]


def _make_anomaly_data(n=200, shift=15.0, seed=42):
    """Generate univariate test data with a known anomaly segment."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 1)))
    X.iloc[80:100] += shift
    return X


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
@pytest.mark.parametrize("Detector", SEGMENT_ANOMALY_DETECTORS)
def test_anomaly_detector_predict_returns_intervals(Detector):
    """All segment anomaly detectors should return DataFrame with Interval ilocs."""
    X = _make_anomaly_data()
    detector = Detector.create_test_instance()
    result = detector.fit_predict(X)

    assert isinstance(result, pd.DataFrame)
    assert "ilocs" in result.columns
    if len(result) > 0:
        assert hasattr(result["ilocs"].iloc[0], "left")
        assert hasattr(result["ilocs"].iloc[0], "right")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
@pytest.mark.parametrize("Detector", TRANSFORM_TESTABLE_DETECTORS)
def test_anomaly_detector_transform_returns_dense(Detector):
    """All segment anomaly detectors' transform should return dense labels."""
    X = _make_anomaly_data()
    detector = Detector.create_test_instance()
    detector.fit(X)
    labels = detector.transform(X)

    assert isinstance(labels, pd.DataFrame)
    assert len(labels) == len(X)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="Test only runs when detection module has changed",
)
def test_stat_threshold_anomaliser_stat_lower_above_upper():
    """Test StatThresholdAnomaliser rejects stat_lower > stat_upper."""
    with pytest.raises(ValueError, match="must be less than or equal to stat_upper"):
        StatThresholdAnomaliser(
            change_detector=MovingWindow(bandwidth=3),
            stat_lower=0.5,
            stat_upper=0.4,
        )
