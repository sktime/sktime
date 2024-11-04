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
        n_jobs=n_jobs,
        sparse_matrix_threshold=sparse_matrix_threshold,
    )
    st_dbscan.fit(X)
    y_pred = st_dbscan.labels_

    assert len(np.unique(y_pred)) == 3

    # rename y_pred labels to match y_true
    y_pred_renamed = y_pred.copy()
    y_pred_renamed[y_pred == 0] = 1
    y_pred_renamed[y_pred == 1] = 2
    y_pred_renamed[y_pred == 2] = 0
    assert np.all(y_pred_renamed == y_true)
