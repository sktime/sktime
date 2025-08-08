"""Tests for the BaseDetector class."""

__author__ = ["Alex-JG3"]
__all__ = []

import pandas as pd
import pytest
from pandas import testing

from sktime.detection.base import BaseDetector
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.xfail(reason="Changed interface, need to update test")
@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "y_sparse, y_dense_expected, index",
    [
        (pd.Series([1, 3]), pd.Series([0, 1, 0, 1]), pd.RangeIndex(0, 4, 1)),
        (pd.Series([1, 3]), pd.Series([0, 1, 0, 1, 0, 0]), pd.RangeIndex(0, 6, 1)),
        (
            pd.Series(
                [1, 2],
                index=pd.IntervalIndex.from_arrays([0, 3], [3, 5], closed="left"),
            ),
            pd.Series([1, 1, 1, 2, 2, -1]),
            pd.RangeIndex(0, 6, 1),
        ),
        (
            pd.Series(
                [1, 2],
                index=pd.IntervalIndex.from_arrays([0, 3], [3, 5], closed="left"),
            ),
            pd.Series([1, 1, 1, 2, 2, -1, -1]),
            pd.RangeIndex(0, 7, 1),
        ),
        (
            pd.Series(
                [1, 2],
                index=pd.IntervalIndex.from_arrays([2, 4], [3, 6], closed="left"),
            ),
            pd.Series([-1, -1, 1, -1, 2, 2, -1]),
            pd.RangeIndex(0, 7, 1),
        ),
    ],
)
def test_sparse_to_dense(y_sparse, y_dense_expected, index):
    """Test converting from sparse to dense."""
    y_dense_actual = BaseDetector.sparse_to_dense(y_sparse, index=index)
    testing.assert_series_equal(y_dense_actual, y_dense_expected)


@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "y_dense, y_sparse_expected",
    [
        (pd.Series([0, 1, 0, 1]), pd.Series([1, 3])),
        (pd.Series([0, 1, 0, 1, 0, 0]), pd.Series([1, 3])),
        (
            pd.Series([-1, -1, -1, 1, 1, -1, 2]),
            pd.Series(
                [1, 2],
                index=pd.IntervalIndex.from_arrays([3, 6], [5, 6], closed="left"),
            ),
        ),
    ],
)
def test_dense_to_sparse(y_dense, y_sparse_expected):
    """Test converting from dense to sparse."""
    y_sparse_actual = BaseDetector.dense_to_sparse(y_dense)
    testing.assert_series_equal(y_sparse_actual, y_sparse_expected)


@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "change_points, expected_segments, start, end",
    [
        (
            pd.Series([1, 2, 5]),
            pd.Series(
                [0, 1, 2, 3],
                index=pd.IntervalIndex.from_breaks([-42, 1, 2, 5, 7], closed="left"),
            ),
            -42,
            7,
        ),
        (
            pd.Series([1, 2, 5]),
            pd.Series(
                [0, 1, 2, 3],
                index=pd.IntervalIndex.from_breaks([0, 1, 2, 5, 7], closed="left"),
            ),
            None,
            7,
        ),
        (
            pd.Series([1, 2, 5]),
            pd.Series(
                [0, 1, 2, 3],
                index=pd.IntervalIndex.from_breaks([0, 1, 2, 5, 6], closed="left"),
            ),
            None,
            None,
        ),
    ],
)
def test_change_points_to_segments(change_points, expected_segments, start, end):
    """Test converting change points to segments."""
    actual_segments = BaseDetector.change_points_to_segments(change_points, start, end)
    testing.assert_series_equal(actual_segments, expected_segments)


@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "segments, expected_change_points",
    [
        (
            pd.DataFrame(
                {
                    "ilocs": pd.IntervalIndex.from_tuples(
                        [
                            (0, 3),
                            (3, 4),
                            (4, 5),
                            (5, 6),
                            (6, 7),
                            (7, 8),
                            (8, 10),
                            (10, 11),
                            (11, 12),
                            (12, 20),
                        ]
                    ),
                    "labels": [0, 2, 1, 0, 2, 1, 0, 2, 1, 0],
                }
            ),
            pd.Index([0, 3, 4, 5, 6, 7, 8, 10, 11, 12]),
        )
    ],
)
def test_segments_to_change_points(segments, expected_change_points):
    """Test converting change points to segments."""
    actual_change_points = BaseDetector.segments_to_change_points(segments)
    assert actual_change_points.equals(expected_change_points)


@pytest.mark.xfail(reason="Changed interface, need to update test")
@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "y_sparse, index, y_dense_expected",
    [
        (
            pd.Series(
                [1, 2, 1],
                index=pd.IntervalIndex.from_arrays([0, 2, 4], [1, 3, 5]),
            ),
            [0, 1, 2, 3, 4, 5, 6],
            pd.Series([-1, 1, -1, 2, -1, 1, -1]),
        )
    ],
)
def test_sparse_segments_to_dense(y_sparse, index, y_dense_expected):
    y_dense_actual = BaseDetector._sparse_segments_to_dense(y_sparse, index)
    testing.assert_series_equal(y_dense_expected, y_dense_actual)


@pytest.mark.skipif(
    not run_test_for_class(BaseDetector),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "y_sparse, index, y_dense_expected",
    [(pd.Series([2, 4]), [0, 1, 2, 3, 4, 5, 6], pd.Series([0, 0, 1, 0, 1, 0, 0]))],
)
def test_sparse_points_to_dense(y_sparse, index, y_dense_expected):
    y_dense_actual = BaseDetector._sparse_points_to_dense(y_sparse, index)
    testing.assert_series_equal(y_dense_actual, y_dense_expected)
