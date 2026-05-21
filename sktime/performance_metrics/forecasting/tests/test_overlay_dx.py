#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for OverlayDX metric."""

import numpy as np
import pandas as pd
import pytest

from sktime.performance_metrics.forecasting import OverlayDX, overlay_dx_score


def test_overlay_dx_perfect_prediction():
    """Test OverlayDX with perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    metric = OverlayDX()
    score = metric(y_true, y_pred)

    # Perfect predictions should give a high score (close to 100)
    assert score > 90.0, f"Expected score > 90 for perfect predictions, got {score}"


def test_overlay_dx_poor_prediction():
    """Test OverlayDX with poor predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    metric = OverlayDX()
    score = metric(y_true, y_pred)

    # Poor predictions should give a low score
    assert score < 50.0, f"Expected score < 50 for poor predictions, got {score}"


def test_overlay_dx_functional_interface():
    """Test the functional interface overlay_dx_score."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0, 2.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0, 1.25])

    score = overlay_dx_score(y_true, y_pred)

    # Should return a numeric score
    assert isinstance(score, (int, float, np.number))
    assert 0.0 <= score <= 100.0


def test_overlay_dx_multivariate():
    """Test OverlayDX with multivariate data."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_pred = np.array([[1.1, 2.1], [2.9, 4.1], [5.1, 5.9]])

    metric = OverlayDX(multioutput="uniform_average")
    score = metric(y_true, y_pred)

    assert isinstance(score, (int, float, np.number))
    assert 0.0 <= score <= 100.0


def test_overlay_dx_multivariate_raw_values():
    """Test OverlayDX with multivariate data returning raw values."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_pred = np.array([[1.1, 2.1], [2.9, 4.1], [5.1, 5.9]])

    metric = OverlayDX(multioutput="raw_values")
    scores = metric(y_true, y_pred)

    # Should return scores for each output
    assert len(scores) == 2
    assert all(0.0 <= s <= 100.0 for s in scores)


def test_overlay_dx_with_pandas():
    """Test OverlayDX with pandas Series."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = pd.Series([1.1, 2.1, 2.9, 4.1, 4.9])

    metric = OverlayDX()
    score = metric(y_true, y_pred)

    assert isinstance(score, (int, float, np.number))
    assert 0.0 <= score <= 100.0


def test_overlay_dx_parameters():
    """Test OverlayDX with custom parameters."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

    # Test with custom tolerance range
    metric = OverlayDX(
        min_percentage=0.5,
        max_percentage=50.0,
        step=0.5
    )
    score = metric(y_true, y_pred)

    assert isinstance(score, (int, float, np.number))
    assert 0.0 <= score <= 100.0


def test_overlay_dx_greater_is_better():
    """Test that OverlayDX has greater_is_better=True."""
    metric = OverlayDX()
    assert metric.greater_is_better is True


def test_overlay_dx_zero_range():
    """Test OverlayDX with constant values (zero range)."""
    y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

    metric = OverlayDX()
    score = metric(y_true, y_pred)

    # Should handle zero range gracefully
    assert isinstance(score, (int, float, np.number))


def test_overlay_dx_negative_values():
    """Test OverlayDX with negative values."""
    y_true = np.array([-5.0, -3.0, -1.0, 1.0, 3.0])
    y_pred = np.array([-4.8, -3.2, -0.9, 1.1, 2.9])

    metric = OverlayDX()
    score = metric(y_true, y_pred)

    assert isinstance(score, (int, float, np.number))
    assert 0.0 <= score <= 100.0


@pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
def test_overlay_dx_multioutput_parameter(multioutput):
    """Test OverlayDX with different multioutput settings."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_pred = np.array([[1.1, 2.1], [2.9, 4.1], [5.1, 5.9]])

    metric = OverlayDX(multioutput=multioutput)
    result = metric(y_true, y_pred)

    if multioutput == "uniform_average":
        assert isinstance(result, (int, float, np.number))
    else:  # raw_values
        assert len(result) == 2


def test_overlay_dx_weighted_multioutput():
    """Test OverlayDX with weighted multioutput."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_pred = np.array([[1.1, 2.1], [2.9, 4.1], [5.1, 5.9]])

    weights = [0.3, 0.7]
    metric = OverlayDX(multioutput=weights)
    score = metric(y_true, y_pred)

    assert isinstance(score, (int, float, np.number))
    assert 0.0 <= score <= 100.0
