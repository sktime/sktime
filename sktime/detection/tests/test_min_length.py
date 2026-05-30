"""Tests for minimum-length segment smoothing."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sktime.detection.base import BaseDetector
from sktime.detection.compose import EnsureMinLengthSegments
from sktime.detection.dummy import ZeroChangePoints, ZeroSegments
from sktime.tests.test_switch import run_test_module_changed

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)


class _FixedSegmentsDetector(BaseDetector):
    _tags = {
        "fit_is_empty": True,
        "task": "segmentation",
        "learning_type": "unsupervised",
    }

    def __init__(self, segments):
        self.segments = segments
        super().__init__()

    def _predict(self, X):
        return self.segments.copy()


class _FixedChangePointsDetector(BaseDetector):
    _tags = {
        "fit_is_empty": True,
        "task": "change_point_detection",
        "learning_type": "unsupervised",
    }

    def __init__(self, change_points):
        self.change_points = change_points
        super().__init__()

    def _predict(self, X):
        return self.change_points.copy()


def _make_X(n_timepoints=10):
    return pd.DataFrame({"value": range(n_timepoints)})


def test_ensure_min_length_segments_greedy_segmentation():
    """Too-short segments are merged according to the forward greedy strategy."""
    segments = pd.DataFrame(
        {
            "ilocs": pd.IntervalIndex.from_tuples(
                [(0, 2), (2, 3), (3, 5), (5, 6), (6, 10)],
                closed="left",
            ),
            "labels": [0, 1, 1, 2, 2],
        }
    )
    detector = _FixedSegmentsDetector(segments=segments)
    smoother = EnsureMinLengthSegments(detector=detector, min_length=3)

    actual = smoother.fit_predict(_make_X())
    expected = pd.DataFrame(
        {
            "ilocs": pd.IntervalIndex.from_tuples(
                [(0, 3), (3, 6), (6, 10)],
                closed="left",
            ),
            "labels": [0, 1, 2],
        }
    )

    assert_frame_equal(actual, expected)


def test_ensure_min_length_segments_merges_short_final_segment_left():
    """A short final segment is merged into its predecessor."""
    segments = pd.DataFrame(
        {
            "ilocs": pd.IntervalIndex.from_tuples(
                [(0, 4), (4, 5)],
                closed="left",
            ),
            "labels": [0, 1],
        }
    )
    detector = _FixedSegmentsDetector(segments=segments)
    smoother = EnsureMinLengthSegments(detector=detector, min_length=2)

    actual = smoother.fit_predict(_make_X(n_timepoints=5))
    expected = pd.DataFrame(
        {
            "ilocs": pd.IntervalIndex.from_tuples([(0, 5)], closed="left"),
            "labels": [0],
        }
    )

    assert_frame_equal(actual, expected)


def test_ensure_min_length_segments_change_points_round_trip():
    """Wrapped change point detectors return smoothed change points."""
    change_points = pd.DataFrame({"ilocs": [2, 3, 7]})
    detector = _FixedChangePointsDetector(change_points=change_points)
    smoother = EnsureMinLengthSegments(detector=detector, min_length=3)

    actual_points = smoother.fit_predict(_make_X())
    expected_points = pd.DataFrame({"ilocs": [3, 7]})
    assert_frame_equal(actual_points, expected_points)

    actual_segments = smoother.predict_segments(_make_X())
    expected_segments = pd.DataFrame(
        {
            "ilocs": pd.IntervalIndex.from_tuples(
                [(0, 3), (3, 7), (7, 10)],
                closed="left",
            )
        }
    )
    assert_frame_equal(actual_segments, expected_segments)


def test_ensure_min_length_segments_requires_contiguous_segments():
    """Segment smoothing rejects gappy segment outputs."""
    segments = pd.DataFrame(
        {
            "ilocs": pd.IntervalIndex.from_tuples(
                [(0, 2), (3, 5)],
                closed="left",
            ),
            "labels": [0, 1],
        }
    )
    detector = _FixedSegmentsDetector(segments=segments)
    smoother = EnsureMinLengthSegments(detector=detector, min_length=2)

    with pytest.raises(ValueError, match="contiguous segment partition"):
        smoother.fit_predict(_make_X(n_timepoints=5))


def test_ensure_min_length_segments_handles_empty_change_points():
    """Empty change-point output stays empty after smoothing."""
    smoother = EnsureMinLengthSegments(detector=ZeroChangePoints(), min_length=3)

    actual = smoother.fit_predict(_make_X())
    expected = pd.DataFrame({"ilocs": pd.Series(dtype="int64")})

    assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "kwargs, expected_exception, expected_match",
    [
        (
            {"detector": ZeroSegments(), "min_length": 0},
            ValueError,
            "`min_length` must be at least 1",
        ),
        (
            {"detector": ZeroSegments(), "strategy": "unknown"},
            ValueError,
            "`strategy` must be one of",
        ),
        (
            {"detector": object()},
            TypeError,
            "`detector` must be an sktime detector",
        ),
    ],
)
def test_ensure_min_length_segments_validates_init_inputs(
    kwargs, expected_exception, expected_match
):
    """Constructor rejects unsupported inputs with clear errors."""
    with pytest.raises(expected_exception, match=expected_match):
        EnsureMinLengthSegments(**kwargs)
