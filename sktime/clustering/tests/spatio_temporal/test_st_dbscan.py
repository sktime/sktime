"""Tests for spatio-temporal DBSCAN."""

import numpy as np
import pytest

from sktime.clustering.spatio_temporal import STDBSCAN
from sktime.clustering.utils.toy_data_generation._make_moving_blobs import (
    make_moving_blobs,
)


@pytest.mark.parametrize(
    "n_times,sparse_matrix_threshold,n_jobs",
    [(3, 2000, -1), (20, 2000, None), (3, 10, 1), (3, 10, -1), (20, 10, None)],
)
def test_st_dbscan(n_times, sparse_matrix_threshold, n_jobs):
    """Test implementation of spatio-temporal DBSCAN."""
    X, y_true = make_moving_blobs(n_times=n_times)

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

    assert len(np.unique(y_pred)) == 3

    # rename y_pred labels to match y_true
    y_pred_renamed = y_pred.copy()
    if n_times == 3:
        y_pred_renamed[y_pred == 0] = 2
        y_pred_renamed[y_pred == 1] = 1
        y_pred_renamed[y_pred == 2] = 0
    elif n_times == 20:
        y_pred_renamed[y_pred == 0] = 0
        y_pred_renamed[y_pred == 1] = 2
        y_pred_renamed[y_pred == 2] = 1
    assert np.all(y_pred_renamed == y_true)


@pytest.mark.parametrize(
    "n_times,frame_size,frame_overlap",
    [(40, 40, 5), (40, 10, 5), (40, 20, 10), (20, 10, None)],
)
def test_st_dbsacan_frame_split(n_times, frame_size, frame_overlap):
    X, y_true = make_moving_blobs(n_times=n_times, cluster_std=0.05)

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

    assert len(np.unique(y_pred)) == 3
    # rename y_pred labels to match y_true
    y_pred_renamed = y_pred.copy()
    if n_times == 20:
        y_pred_renamed[y_pred == 0] = 0
        y_pred_renamed[y_pred == 1] = 2
        y_pred_renamed[y_pred == 2] = 1
    elif n_times == 40:
        y_pred_renamed[y_pred == 0] = 1
        y_pred_renamed[y_pred == 1] = 0
        y_pred_renamed[y_pred == 2] = 2
    assert np.all(y_pred_renamed == y_true)


@pytest.mark.parametrize(
    "frame_size,sparse_matrix_threshold",
    [(None, 2000), (None, 10), (20, 2000), (20, 10)],
)
def test_st_dbscan_data_with_noise(frame_size, sparse_matrix_threshold):
    X, y_true = make_moving_blobs(
        n_times=20, cluster_std=1, centers_origin=[[-10, -10], [0, 0], [10, 10]]
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

    assert len(np.unique(y_pred)) == 4

    # rename y_pred labels to match y_true
    y_pred_renamed = y_pred.copy()
    y_pred_renamed[y_pred == 0] = 0
    y_pred_renamed[y_pred == 1] = 2
    y_pred_renamed[y_pred == 2] = 1

    assert np.all(y_pred_renamed[y_pred != -1] == y_true[y_pred != -1])
