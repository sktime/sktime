"""Tests for CircularBinarySegmentation."""

import pytest

from sktime.detection._skchange.anomaly_detectors import CircularBinarySegmentation
from sktime.detection._skchange.change_scores import ChangeScore
from sktime.detection._skchange.costs import COSTS


def test_invalid_change_scores():
    """
    Test that CircularBinarySegmentation raises an error when given an invalid score.
    """
    with pytest.raises(ValueError, match="anomaly_score"):
        CircularBinarySegmentation("l2")
    with pytest.raises(ValueError, match="anomaly_score"):
        CircularBinarySegmentation(ChangeScore(COSTS[2].create_test_instance()))
