"""Basic tests for all change detectors."""

import pandas as pd
import pytest

from sktime.detection._skchange.change_detectors import CHANGE_DETECTORS
from sktime.detection._skchange.change_detectors.base import BaseChangeDetector
from sktime.detection._skchange.datasets import generate_alternating_data

n_segments = 3
seg_len = 50
changepoint_data = generate_alternating_data(
    n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
)


@pytest.mark.parametrize("Estimator", CHANGE_DETECTORS)
def test_change_detector_predict(Estimator):
    """Test changepoint detector predict (sparse output)."""
    detector = Estimator.create_test_instance()
    changepoints = detector.fit_predict(changepoint_data)["ilocs"]
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len


@pytest.mark.parametrize("Estimator", CHANGE_DETECTORS)
def test_change_detector_transform(Estimator: BaseChangeDetector):
    """Test changepoint detector transform (dense output)."""
    detector = Estimator.create_test_instance()
    labels: pd.Series = detector.fit_transform(changepoint_data)["labels"]

    assert labels.nunique() == n_segments
    assert labels[seg_len - 1] == 0.0 and labels[seg_len] == 1.0
