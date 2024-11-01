"""Tests for spatio-temporal DBSCAN."""

import numpy as np
from sklearn.datasets import make_blobs

from sktime.clustering.spatio_temporal import STDBSCAN


def spatio_temporal_data(n_times=3, n_samples=15, cluster_std=0.20, random_state=10):
    """Generate spatio-temporal data."""
    centers = np.array([[0, 0], [1, 1], [-1, -1]], dtype=float)
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    for t in range(n_times - 1):
        centers += 0.1
        _X, _y_true = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        X = np.vstack((X, _X))
        y_true = np.hstack((y_true, _y_true))
    # add time column
    time = np.arange(n_times).repeat(n_samples)
    X = np.column_stack([time, X])
    return X, y_true


def test_st_dbscan():
    """Test implementation of spatio-temporal DBSCAN."""
    X, y_true = spatio_temporal_data()

    st_dbscan = STDBSCAN(eps1=0.5, eps2=3, min_samples=5, metric="euclidean", n_jobs=1)
    st_dbscan.fit(X)
    y_pred = st_dbscan.labels_

    assert len(np.unique(y_pred)) == 3

    # rename y_pred labels to match y_true
    y_pred_renamed = y_pred.copy()
    y_pred_renamed[y_pred == 0] = 1
    y_pred_renamed[y_pred == 1] = 2
    y_pred_renamed[y_pred == 2] = 0
    assert np.all(y_pred_renamed == y_true)
