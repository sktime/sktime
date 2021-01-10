# -*- coding: utf-8 -*-
import pytest
from sktime.distances.elastic_cython import dtw_distance
from sktime.clustering.utils import (
    Data_Frame,
)
from sktime.clustering._cluster import Cluster
from sklearn.cluster import KMeans


def test_cluster(df_x: Data_Frame):
    try:
        Cluster(
            model=KMeans(
                n_clusters=3,
                init="random",
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0,
            ),
        )
    except Exception:
        pytest.fail("Failed to construct base Cluster with no parameters")

    try:
        Cluster(
            model=KMeans(
                n_clusters=3,
                init="random",
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0,
            ),
            distance="dtw",
        )
    except Exception:
        pytest.fail("Failed to construct base Cluster with no parameters")

    try:
        Cluster(
            model=KMeans(
                n_clusters=3,
                init="random",
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0,
            ),
            distance=dtw_distance,
        )
    except Exception:
        pytest.fail("Failed to construct base Cluster with no parameters")

    # sklearn_train_data: SkLearn_Data = convert_df_to_sklearn_format(df_x)
    # km = KMeans(
    #     n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04,
    # random_state=0)
    # km.fit(sklearn_train_data)
