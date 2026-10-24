"""Tests for ClusterSegmenter."""

import numpy as np
import pandas as pd
import pytest

from sktime.detection.clust import ClusterSegmenter
from sktime.tests.test_switch import run_test_for_class


def _kmeans(n_clusters=2):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_clusters, random_state=0, n_init=10)


@pytest.mark.skipif(
    not run_test_for_class(ClusterSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cluster_segmenter_returns_labelled_segments():
    """predict returns segments covering the series, with one label per segment."""
    X = pd.DataFrame({"a": [1.0, 1.1, 0.9, 1.0, 10.0, 10.2, 9.8, 10.1]})
    segmenter = ClusterSegmenter(_kmeans())
    y = segmenter.fit_predict(X)

    assert list(y.columns) == ["ilocs", "labels"]
    assert [iv.left for iv in y["ilocs"]] == [0, 4]
    assert [iv.right for iv in y["ilocs"]] == [4, 8]
    assert y["labels"].iloc[0] != y["labels"].iloc[1]

    # derived outputs are consistent with the segments
    assert list(segmenter.predict_points(X)["ilocs"]) == [0, 4]
    labels = segmenter.transform(X).iloc[:, 0].to_numpy()
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1
    assert labels[0] != labels[-1]


@pytest.mark.skipif(
    not run_test_for_class(ClusterSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cluster_segmenter_multivariate():
    """Time points of multivariate series are clustered as vectors."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        np.r_[rng.normal(0, 0.1, (10, 2)), rng.normal(5, 0.1, (10, 2))],
        columns=["a", "b"],
    )
    y = ClusterSegmenter(_kmeans()).fit_predict(X)

    assert [iv.left for iv in y["ilocs"]] == [0, 10]
    assert y["ilocs"].iloc[-1].right == 20


@pytest.mark.skipif(
    not run_test_for_class(ClusterSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cluster_segmenter_series_input():
    """pd.Series input is accepted and segmented like a single column frame."""
    X = pd.Series(np.r_[np.zeros(5), np.ones(5) * 10])
    y = ClusterSegmenter(_kmeans()).fit_predict(X)

    assert [iv.left for iv in y["ilocs"]] == [0, 5]
    assert y["ilocs"].iloc[-1].right == 10
