"""Tests for MovingWindow and all available scores."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection._skchange.anomaly_scores import LocalAnomalyScore
from sktime.detection._skchange.change_detectors import SeededBinarySegmentation
from sktime.detection._skchange.change_scores import CHANGE_SCORES
from sktime.detection._skchange.costs import COSTS
from sktime.detection._skchange.datasets import generate_alternating_data

SCORES_AND_COSTS = CHANGE_SCORES + COSTS


def test_invalid_parameters():
    """Test invalid input parameters to SeededBinarySegmentation.

    These tests serve as tests for the input validators.
    """
    with pytest.raises(ValueError):
        SeededBinarySegmentation(penalty=-0.1)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(growth_factor=1.0)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(growth_factor=None)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(growth_factor=2.01)


def test_invalid_data():
    """Test invalid input data to SeededBinarySegmentation.

    These tests serve as tests for the input validators.
    """
    detector = SeededBinarySegmentation()
    with pytest.raises(ValueError):
        detector.fit_predict(np.array([1.0]))

    with pytest.raises(ValueError):
        detector.fit_predict(pd.Series([1.0, np.nan, 1.0, 1.0]))


@pytest.mark.parametrize("selection_method", ["greedy", "narrowest"])
def test_selection_method(selection_method):
    """Test SeededBinarySegmentation selection method."""
    n_segments = 2
    seg_len = 10
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=200
    )
    detector = SeededBinarySegmentation.create_test_instance()
    detector.set_params(selection_method=selection_method)
    changepoints = detector.fit_predict(df)["ilocs"]
    assert len(changepoints) == n_segments - 1
    assert changepoints[0] == 10


def test_invalid_selection_method():
    """Test invalid selection method."""
    detector = SeededBinarySegmentation.create_test_instance()
    with pytest.raises(ValueError):
        detector.set_params(selection_method="greedy2")


def test_invalid_change_scores():
    """
    Test that SeededBinarySegmentation raises an error when given an invalid cost.
    """
    with pytest.raises(ValueError, match="change_score"):
        SeededBinarySegmentation("l2")
    with pytest.raises(ValueError, match="change_score"):
        SeededBinarySegmentation(LocalAnomalyScore(COSTS[0].create_test_instance()))
