# -*- coding: utf-8 -*-
"""Test segmentation metrics."""

import pytest

from sktime.performance_metrics.annotation.metrics import (
    annotation_error,
    hausdorff_error,
    prediction_ratio,
)


@pytest.fixture
def exact_match():
    """Change points with exact match."""
    change_points = list(range(5))
    return change_points, change_points


@pytest.fixture
def different_lengths():
    """Change points with different lengths."""
    return list(range(5)), list(range(10))


def test_annotation_error_exact(exact_match):
    """Test metric."""
    cp_true, cp_pred = exact_match
    assert annotation_error(cp_true, cp_pred) == 0.0


def test_hausdorff_error_exact(exact_match):
    """Test metric."""
    cp_true, cp_pred = exact_match
    assert hausdorff_error(cp_true, cp_pred) == 0.0


def test_prediction_ratio_exact(exact_match):
    """Test metric."""
    cp_true, cp_pred = exact_match
    assert prediction_ratio(cp_true, cp_pred) == 1.0


def test_annotation_error(different_lengths):
    """Test metric."""
    cp_true, cp_pred = different_lengths
    assert annotation_error(cp_true, cp_pred) == 5.0


def test_hausdorff_error(different_lengths):
    """Test metric."""
    cp_true, cp_pred = different_lengths
    assert hausdorff_error(cp_true, cp_pred) == 5.0


def test_prediction_ratio(different_lengths):
    """Test metric."""
    cp_true, cp_pred = different_lengths
    assert prediction_ratio(cp_true, cp_pred) == 2.0
