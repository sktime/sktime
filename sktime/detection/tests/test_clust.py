"""Tests for ClusterSegmenter."""

import pandas as pd
import pytest
from sklearn.cluster import KMeans

from sktime.detection.clust import ClusterSegmenter
from sktime.tests.test_switch import run_test_for_class

pytestmark = pytest.mark.skipif(
    not run_test_for_class(ClusterSegmenter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)


def test_cluster_segmenter_predict_returns_sparse_segments():
    """Predict should return labelled intervals for contiguous clusters."""
    X = pd.DataFrame({"x": [0, 0, 0, 10, 10, 10]})
    est = ClusterSegmenter(clusterer=KMeans(n_clusters=2, random_state=0)).fit(X)

    y_pred = est.predict(X)

    assert list(y_pred.columns) == ["ilocs", "labels"]
    assert pd.IntervalIndex(y_pred["ilocs"]).equals(
        pd.IntervalIndex.from_tuples([(0, 3), (3, 6)], closed="left")
    )
    assert y_pred["labels"].nunique() == 2


def test_cluster_segmenter_predict_points_returns_boundaries():
    """Predict points should convert the segmentation into segment boundaries."""
    X = pd.DataFrame({"x": [0, 0, 0, 10, 10, 10]})
    est = ClusterSegmenter(clusterer=KMeans(n_clusters=2, random_state=0)).fit(X)

    y_pred = est.predict_points(X)

    assert y_pred["ilocs"].tolist() == [0, 3]


def test_cluster_segmenter_transform_returns_dense_labels():
    """Transform should return one label per time point."""
    X = pd.DataFrame({"x": [0, 0, 0, 10, 10, 10]})
    est = ClusterSegmenter(clusterer=KMeans(n_clusters=2, random_state=0)).fit(X)

    y_pred = est.transform(X)

    assert list(y_pred.columns) == ["labels"]
    assert len(y_pred) == len(X)
    assert len(set(y_pred["labels"].iloc[:3])) == 1
    assert len(set(y_pred["labels"].iloc[3:])) == 1
    assert y_pred["labels"].iloc[0] != y_pred["labels"].iloc[3]
