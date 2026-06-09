"""Basic tests for all change detectors."""

import pandas as pd
import pytest

from sktime._vendor.skchange.change_detectors import CHANGE_DETECTORS
from sktime._vendor.skchange.change_detectors.base import BaseChangeDetector
from sktime._vendor.skchange.datasets import generate_alternating_data

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
    # platform-independent: tests proximity to expected position instead of exact index
    assert len(changepoints) == n_segments - 1 and abs(changepoints[0] - seg_len) <= 5


@pytest.mark.parametrize("Estimator", CHANGE_DETECTORS)
def test_change_detector_transform(Estimator: BaseChangeDetector):
    """Test changepoint detector transform (dense output)."""
    detector = Estimator.create_test_instance()
    labels: pd.Series = detector.fit_transform(changepoint_data)["labels"]

    assert labels.nunique() == n_segments
    # platform-independent: tests label transition near boundary instead of exact index
    transition_found = any(
        labels[i] != labels[i + 1]
        for i in range(max(0, seg_len - 5), min(len(labels) - 1, seg_len + 5))
    )
    assert transition_found
