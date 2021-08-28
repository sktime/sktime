# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering.tests._clustering_tests import generate_univaritate_series
from sktime.clustering.partitioning._center_initializers import (
    ForgyCenterInitializer,
    RandomCenterInitializer,
)
from sktime.clustering._k_medoids import TimeSeriesKMedoids
from sktime.clustering._k_means import TimeSeriesKMeans


def test_forgy_cluster_center_initializer():
    rng = np.random.RandomState(0)
    X = generate_univaritate_series(n=20, size=1, rng=rng, dtype=np.float32)
    forgy_centers = ForgyCenterInitializer(X, 5, random_state=rng)
    centers = forgy_centers.initialize_centers()
    assert np.array_equal(
        np.array(
            [[0.33367434], [1.4940791], [0.95008844], [0.4001572], [-0.10321885]],
            dtype=np.float32,
        ),
        centers,
    )


def test_random_cluster_center_initializer():
    n_clusters = 3
    k_medians = TimeSeriesKMedoids(n_clusters=n_clusters)
    rng = np.random.RandomState(0)
    X = generate_univaritate_series(n=20, size=n_clusters, rng=rng, dtype=np.int64)

    random_centers_medians = RandomCenterInitializer(
        X, n_clusters, k_medians.calculate_new_centers, rng
    )
    centers = random_centers_medians.initialize_centers()

    assert np.array_equal(
        np.array([[472, 600, 396], [544, 543, 714], [684, 559, 629]]), centers
    )

    k_means = TimeSeriesKMeans(n_clusters=n_clusters)

    random_centers_mean = RandomCenterInitializer(
        X, n_clusters, k_means.calculate_new_centers, rng
    )
    centers = random_centers_mean.initialize_centers()

    assert np.array_equal(
        np.array([[520, 521, 692], [491, 581, 409], [695, 492, 403]]), centers
    )

    k_means_dtw = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")

    random_centers_mean = RandomCenterInitializer(
        X, n_clusters, k_means_dtw.calculate_new_centers, rng
    )
    centers = random_centers_mean.initialize_centers()

    assert np.array_equal(
        np.array([[516, 564, 670], [564, 546, 480], [632, 438, 527]]), centers
    )
