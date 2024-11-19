"""Tests for spatio-temporal DBSCAN."""

import numpy as np
import pytest

from sktime.clustering.spatio_temporal import STDBSCAN
from sktime.clustering.utils.toy_data_generation._make_moving_blobs import (
    make_moving_blobs,
)


def _rename_labels(X, centers_origin, y_pred, n_labels):
    """Rename cluster labels to match the true labels."""
    # rename y_pred labels to match y_true
    y_pred_renamed = y_pred.copy()

    # find the closest cluster center at time 0
    for l in range(n_labels):
        # data at index 0
        t0_mask = X.index.get_level_values(1).to_numpy() == 0
        X_t0 = X[t0_mask].to_numpy()
        y_pred_t0 = y_pred[t0_mask]
        c = np.mean(X_t0[y_pred_t0 == l], axis=0)

        # find the closest cluster center at time 0
        dist = np.linalg.norm(centers_origin - c[1:], axis=1)

        # rename cluster l to the closest cluster center at time 0
        y_pred_renamed[y_pred == l] = np.argmin(dist)

    return y_pred_renamed


@pytest.mark.parametrize(
    "n_times,sparse_matrix_threshold,n_jobs",
    [(3, 2000, -1), (20, 2000, None), (3, 10, 1), (3, 10, -1), (20, 10, None)],
)
def test_st_dbscan(n_times, sparse_matrix_threshold, n_jobs):
    """Test implementation of spatio-temporal DBSCAN."""
    centers_origin = np.array([[-1, -1], [0, 0], [1, 1]])
    X, y_true = make_moving_blobs(n_times=n_times, centers_origin=centers_origin)

    st_dbscan = STDBSCAN(
        eps1=0.5,
        eps2=3,
        min_samples=5,
        metric="euclidean",
        frame_size=None,
        frame_overlap=None,
        sparse_matrix_threshold=sparse_matrix_threshold,
        n_jobs=n_jobs,
    )
    st_dbscan.fit(X)
    y_pred = st_dbscan.labels_

    n_labels = len(np.unique(y_pred[y_pred != -1]))
    assert n_labels == 3
    y_pred_renamed = _rename_labels(X, centers_origin, y_pred, n_labels)
    assert np.all(y_pred_renamed == y_true)


def test_st_dbscan_data_not_sorted_by_time():
    centers_origin = np.array([[-1, -1], [0, 0], [1, 1]])
    X, y_true = make_moving_blobs(n_times=10, centers_origin=centers_origin)

    X_wrong_sorting = X.sort_index(inplace=False, ascending=False, level=1)

    st_dbscan = STDBSCAN(eps1=0.5, eps2=3, min_samples=5)
    st_dbscan.fit(X_wrong_sorting)
    y_pred = st_dbscan.labels_
    n_labels = len(np.unique(y_pred[y_pred != -1]))
    assert n_labels == 3
    y_pred_renamed = _rename_labels(X, centers_origin, y_pred, n_labels)
    assert np.all(y_pred_renamed == y_true)


@pytest.mark.parametrize(
    "n_times,frame_size,frame_overlap",
    [(40, 40, 5), (40, 10, 5), (40, 20, 10), (20, 10, None)],
)
def test_st_dbscan_frame_split(n_times, frame_size, frame_overlap):
    centers_origin = np.array([[-1, -1], [0, 0], [1, 1]])
    X, y_true = make_moving_blobs(
        n_times=n_times, cluster_std=0.05, centers_origin=centers_origin
    )

    st_dbscan = STDBSCAN(
        eps1=0.5,
        eps2=3,
        min_samples=5,
        frame_size=20,
        frame_overlap=5,
        n_jobs=None,
    )

    st_dbscan.fit(X)

    y_pred = st_dbscan.labels_

    n_labels = len(np.unique(y_pred[y_pred != -1]))
    assert n_labels == 3
    y_pred_renamed = _rename_labels(X, centers_origin, y_pred, n_labels)
    assert np.all(y_pred_renamed == y_true)


@pytest.mark.parametrize(
    "frame_size,sparse_matrix_threshold",
    [(None, 2000), (None, 10), (20, 2000), (20, 10)],
)
def test_st_dbscan_data_with_noise(frame_size, sparse_matrix_threshold):
    centers_origin = np.array([[-10, -10], [0, 0], [10, 10]])
    X, y_true = make_moving_blobs(
        n_times=20, cluster_std=1, centers_origin=centers_origin
    )

    st_dbscan = STDBSCAN(
        eps1=1,
        eps2=3,
        min_samples=5,
        metric="euclidean",
        frame_size=frame_size,
        frame_overlap=None,
        sparse_matrix_threshold=sparse_matrix_threshold,
        n_jobs=None,
    )

    st_dbscan.fit(X)
    y_pred = st_dbscan.labels_

    n_labels = len(np.unique(y_pred[y_pred != -1]))
    assert n_labels == 3

    y_pred_renamed = _rename_labels(X, centers_origin, y_pred, n_labels)

    assert np.all(y_pred_renamed[y_pred != -1] == y_true[y_pred != -1])
