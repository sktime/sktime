"""Tests for WindowSegmenter."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection.wclust import (
    WindowSegmenter,
    _overlapping_window,
    _window_timeseries,
)
from sktime.tests.test_switch import run_test_for_class


def _kmeans(n_clusters=2):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_clusters, random_state=0, n_init=10)


@pytest.mark.skipif(
    not run_test_for_class(WindowSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_windows_keep_multivariate_layout():
    """Windows are numpy3D arrays with the time points of each feature in order."""
    X = pd.DataFrame(np.arange(24).reshape(12, 2), columns=["a", "b"])

    windows = _window_timeseries(4, X)
    assert windows.shape == (3, 2, 4)
    for i in range(3):
        for j in range(2):
            np.testing.assert_array_equal(windows[i, j], X.iloc[4 * i : 4 * i + 4, j])

    windows = _overlapping_window(4, 2, X)
    assert windows.shape == (6, 2, 4)
    for i in range(5):
        for j in range(2):
            np.testing.assert_array_equal(windows[i, j], X.iloc[2 * i : 2 * i + 4, j])
    # the last window is zero padded
    np.testing.assert_array_equal(windows[5, 0], [20, 22, 0, 0])


@pytest.mark.skipif(
    not run_test_for_class(WindowSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("overlap, step_size", [(False, 1), (True, 1), (True, 2)])
def test_window_segmenter_segments_cover_series(overlap, step_size):
    """Segments cover all time points, including the last one, with one label each."""
    X = pd.DataFrame({"a": [1, 1, 1, 1, 10, 10, 10, 10]})
    segmenter = WindowSegmenter(
        clusterer=_kmeans(), window_size=2, overlap=overlap, step_size=step_size
    )
    result = segmenter.fit_predict(X)

    intervals = list(result["ilocs"])
    assert intervals[0].left == 0
    assert intervals[-1].right == len(X)
    for prev, nxt in zip(intervals[:-1], intervals[1:]):
        assert prev.right == nxt.left
    # the two regimes are found
    assert [iv.left for iv in intervals] == [0, 4]
    assert result["cluster"].iloc[0] != result["cluster"].iloc[1]


@pytest.mark.skipif(
    not run_test_for_class(WindowSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("overlap", [False, True])
def test_window_segmenter_return_labels(overlap):
    """With return_segments=False, one label per time point is returned."""
    X = pd.DataFrame({"a": [1, 1, 1, 1, 10, 10, 10, 10]})
    segmenter = WindowSegmenter(
        clusterer=_kmeans(), window_size=2, overlap=overlap, return_segments=False
    )
    result = segmenter.fit_predict(X)

    labels = np.asarray(result).ravel()
    assert len(labels) == len(X)
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[-1]


@pytest.mark.skipif(
    not run_test_for_class(WindowSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_window_segmenter_multivariate_overlap():
    """Overlapping windows of multivariate series are clustered per time point."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        np.r_[rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))],
        columns=["a", "b"],
    )
    segmenter = WindowSegmenter(
        clusterer=_kmeans(), window_size=4, overlap=True, step_size=2
    )
    result = segmenter.fit_predict(X)

    assert [iv.left for iv in result["ilocs"]] == [0, 20]
    assert result["ilocs"].iloc[-1].right == 40
