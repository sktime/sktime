"""Tests for all annotators/detectors in skchange."""

import numpy as np
import pandas as pd
import pytest
from sktime.tests.test_all_estimators import VALID_ESTIMATOR_TAGS

from sktime.detection._skchange.anomaly_detectors import ANOMALY_DETECTORS
from sktime.detection._skchange.base import BaseDetector
from sktime.detection._skchange.change_detectors import CHANGE_DETECTORS
from sktime.detection._skchange.datasets import generate_anomalous_data

ALL_DETECTORS = ANOMALY_DETECTORS + CHANGE_DETECTORS
VALID_DETECTOR_TAGS = list(VALID_ESTIMATOR_TAGS) + [
    "task",
    "learning_type",
    "capability:variable_identification",
]


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_fit(Detector: BaseDetector):
    """Test fit method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    y = pd.Series(np.zeros(len(x)))  # For coverage testing.
    fit_detector = detector.fit(x, y)
    assert issubclass(detector.__class__, BaseDetector)
    assert issubclass(fit_detector.__class__, BaseDetector)
    assert isinstance(fit_detector, Detector)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_predict(Detector: BaseDetector):
    """Test predict method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=60)
    y = detector.fit_predict(x)
    assert isinstance(y, (pd.Series, pd.DataFrame))


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_transform(Detector: BaseDetector):
    """Test transform method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=61)
    y = detector.fit_transform(x)
    assert isinstance(y, (pd.Series, pd.DataFrame))
    assert len(x) == len(y)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_transform_scores(Detector: BaseDetector):
    """Test transform_scores method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=62)
    try:
        y = detector.fit(x).transform_scores(x)
        assert isinstance(y, (pd.Series, pd.DataFrame))
    except NotImplementedError:
        pass


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_update(Detector: BaseDetector):
    """Test update method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    x_train = x.iloc[:20]
    x_next = x[20:]
    detector.fit(x_train)
    detector.update_predict(x_next)
    assert issubclass(detector.__class__, BaseDetector)
    assert isinstance(detector, Detector)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_sparse_to_dense(Detector):
    """Test that predict + sparse_to_dense == transform."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=63)
    detections = detector.fit_predict(x)
    labels = detector.sparse_to_dense(detections, x.index, x.columns)
    labels_transform = detector.fit_transform(x)
    assert labels.equals(labels_transform)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_dense_to_sparse(Detector):
    """Test that transform + dense_to_sparse == predict."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=63)
    labels = detector.fit_transform(x)
    detections = detector.dense_to_sparse(labels)
    detections_predict = detector.fit_predict(x)
    assert detections.equals(detections_predict)


def test_detector_not_implemented_methods():
    detector = BaseDetector()
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")

    detector._is_fitted = True  # Required for the following functions to run
    with pytest.raises(NotImplementedError):
        detector.predict(x)
    with pytest.raises(NotImplementedError):
        detector.transform(x)
    with pytest.raises(NotImplementedError):
        detector.transform_scores(x)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_numpy_input(Detector: BaseDetector):
    """Test transform method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=61)
    y = detector.fit_transform(x.values)
    assert isinstance(y, (pd.Series, pd.DataFrame))
    assert len(x) == len(y)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_change_points_to_segments(Detector: BaseDetector):
    """Test change_points_to_segments method."""
    detector = Detector.create_test_instance()

    # Test with multiple change points
    change_points = pd.DataFrame({"ilocs": pd.Series([2, 5, 8], dtype="int64")})

    # Test with start and end not provided
    segments = detector.change_points_to_segments(change_points)
    assert segments.equals(
        pd.DataFrame(
            {
                "ilocs": pd.IntervalIndex.from_breaks([0, 2, 5, 8, 9], closed="left"),
                "labels": pd.Series([0, 1, 2, 3], dtype="int64"),
            }
        )
    )

    # Test with invalid start
    with pytest.raises(ValueError):
        detector.change_points_to_segments(change_points, start=3, end=10)

    # Test with invalid end
    with pytest.raises(ValueError):
        detector.change_points_to_segments(change_points, start=0, end=7)

    # Test with unsorted change points
    change_points = pd.DataFrame({"ilocs": pd.Series([5, 2, 8], dtype="int64")})
    with pytest.raises(ValueError):
        detector.change_points_to_segments(change_points, start=0, end=10)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_valid_detector_class_tags(Detector: type[BaseDetector]):
    """Check that Detector class tags are in VALID_DETECTOR_TAGS."""
    for tag in Detector.get_class_tags().keys():
        msg = "Found invalid tag: %s" % tag
        assert tag in VALID_DETECTOR_TAGS, msg


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_valid_detector_tags(Detector: type[BaseDetector]):
    """Check that Detector class tags are in VALID_DETECTOR_TAGS."""
    for tag in Detector.get_class_tags().keys():
        msg = "Found invalid tag: %s" % tag
        assert tag in VALID_DETECTOR_TAGS, msg
