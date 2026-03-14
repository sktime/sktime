"""Basic tests for all anomaly detectors."""

import pandas as pd
import pytest

from sktime.detection._skchange.anomaly_detectors import (
    COLLECTIVE_ANOMALY_DETECTORS,
    StatThresholdAnomaliser,
)
from sktime.detection._skchange.anomaly_detectors.base import BaseSegmentAnomalyDetector
from sktime.detection._skchange.change_detectors import MovingWindow
from sktime.detection._skchange.datasets import generate_anomalous_data

true_anomalies = [(30, 35), (70, 75)]
anomaly_data = generate_anomalous_data(
    100, anomalies=true_anomalies, means=[10.0, 15.0], random_state=2
)
anomaly_free_data = generate_anomalous_data(100, random_state=1)


@pytest.mark.parametrize("Estimator", COLLECTIVE_ANOMALY_DETECTORS)
def test_segment_anomaly_detector_predict(Estimator: BaseSegmentAnomalyDetector):
    """Test segment anomaly detector's predict method (sparse output)."""
    detector = Estimator.create_test_instance()
    detector.fit(anomaly_free_data)
    anomalies = detector.predict(anomaly_data)["ilocs"]

    assert len(anomalies) == len(true_anomalies)
    for i, (start, end) in enumerate(true_anomalies):
        assert anomalies.array.left[i] == start and anomalies.array.right[i] == end


@pytest.mark.parametrize("Estimator", COLLECTIVE_ANOMALY_DETECTORS)
def test_segment_anomaly_detector_transform(
    Estimator: BaseSegmentAnomalyDetector,
):
    """Test segment anomaly detector's transform method (dense output)."""
    detector = Estimator.create_test_instance()
    detector.fit(anomaly_free_data)
    labels = detector.transform(anomaly_data)
    true_segment_anomalies = pd.DataFrame(
        {"ilocs": pd.IntervalIndex.from_tuples(true_anomalies, closed="left")}
    )
    true_anomaly_labels = BaseSegmentAnomalyDetector.sparse_to_dense(
        true_segment_anomalies, anomaly_data.index
    )
    labels.equals(true_anomaly_labels)

    # Similar test that does not depend on sparse_to_dense, just to be sure.
    labels = labels.iloc[:, 0]
    assert labels.nunique() == len(true_anomalies) + 1
    for i, (start, end) in enumerate(true_anomalies):
        assert (labels.iloc[start:end] == i + 1).all()


def test_dense_to_sparse_invalid_columns():
    """Test dense_to_sparse method with invalid DataFrame input columns."""
    invalid_df = pd.DataFrame({"invalid_column": [0, 1, 0, 1]})
    with pytest.raises(ValueError):
        BaseSegmentAnomalyDetector.dense_to_sparse(invalid_df)


def test_stat_threshold_anomaliser_raises_stat_lower_above_stat_upper():
    """Test StatThresholdAnomaliser with valid parameters."""
    change_detector = MovingWindow(bandwidth=3)
    with pytest.raises(ValueError, match="must be less than or equal to stat_upper"):
        StatThresholdAnomaliser(
            change_detector=change_detector, stat_lower=0.5, stat_upper=0.4
        )
