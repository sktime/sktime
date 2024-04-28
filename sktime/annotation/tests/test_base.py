"""Tests for the BaseSeriesAnnotator class."""

__author__ = ["Alex-JG3"]
__all__ = []

import pandas as pd
import pytest

from sktime.annotation.base._base import BaseSeriesAnnotator


@pytest.mark.parametrize(
    "y_sparse, y_dense_expected, length",
    [
        (pd.Series([1, 3]), pd.Series([0, 1, 0, 1], dtype="int32"), None),
        (pd.Series([1, 3]), pd.Series([0, 1, 0, 1, 0, 0], dtype="int32"), 6),
        (
            pd.DataFrame({"seg_label": [1, 2], "seg_start": [0, 3], "seg_end": [2, 4]}),
            pd.Series([1, 1, 1, 2, 2], dtype="int32"),
            None,
        ),
        (
            pd.DataFrame({"seg_label": [1, 2], "seg_start": [0, 3], "seg_end": [2, 4]}),
            pd.Series([1, 1, 1, 2, 2, -1, -1], dtype="int32"),
            7,
        ),
        (
            pd.DataFrame({"seg_label": [1, 2], "seg_start": [2, 4], "seg_end": [2, 5]}),
            pd.Series([-1, -1, 1, -1, 2, 2, -1], dtype="int32"),
            7,
        ),
    ],
)
def test_sparse_to_dense(y_sparse, y_dense_expected, length):
    """Test converting from sparse to dense."""
    y_dense_actual = BaseSeriesAnnotator.sparse_to_dense(y_sparse, length=length)
    assert y_dense_actual.equals(y_dense_expected)


@pytest.mark.parametrize(
    "y_dense, y_sparse_expected",
    [
        (pd.Series([0, 1, 0, 1]), pd.Series([1, 3])),
        (pd.Series([0, 1, 0, 1, 0, 0]), pd.Series([1, 3])),
        (
            pd.Series([-1, -1, -1, 1, 1, -1, 2]),
            pd.DataFrame({"seg_label": [1, 2], "seg_start": [3, 6], "seg_end": [4, 6]}),
        ),
    ],
)
def test_dense_to_sparse(y_dense, y_sparse_expected):
    """Test converting from dense to sparse."""
    y_sparse_actual = BaseSeriesAnnotator.dense_to_sparse(y_dense)
    assert y_sparse_actual.equals(y_sparse_expected)


@pytest.mark.parametrize(
    "change_points, expected_segments, length",
    [
        (
            pd.Series([1, 2, 5]),
            pd.DataFrame(
                {
                    "seg_label": [1, 2, 3, 4],
                    "seg_start": [0, 1, 2, 5],
                    "seg_end": [0, 1, 4, 9],
                }
            ),
            10,
        )
    ],
)
def test_change_points_to_segments(change_points, expected_segments, length):
    """Test converting change points to segments."""
    actual_segments = BaseSeriesAnnotator.change_points_to_segments(
        change_points, length
    )
    assert actual_segments.equals(expected_segments)


@pytest.mark.parametrize(
    "segments, expected_change_points",
    [
        (
            pd.DataFrame(
                {
                    "seg_label": [1, 2, 3, 4],
                    "seg_start": [0, 1, 2, 5],
                    "seg_end": [0, 1, 4, 9],
                }
            ),
            pd.Series([1, 2, 5]),
        )
    ],
)
def test_segments_to_change_points(segments, expected_change_points):
    """Test converting change points to segments."""
    actual_change_points = BaseSeriesAnnotator.segments_to_change_points(segments)
    assert actual_change_points.equals(expected_change_points)
