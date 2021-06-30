# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering._k_medoids import TimeSeriesKMedoids
from sktime.clustering.tests._clustering_tests import (
    generate_univaritate_series,
    run_clustering_experiment,
)


def test_k_medoids():
    rng = np.random.RandomState(1)
    X_train = generate_univaritate_series(n=100, size=5, rng=rng, dtype=np.double)
    X_test = generate_univaritate_series(
        n=10, size=5, rng=np.random.RandomState(2), dtype=np.double
    )

    clusters, _ = run_clustering_experiment(
        TimeSeriesKMedoids(n_clusters=5, max_iter=50, random_state=rng), X_train, X_test
    )
    assert np.array_equal(np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 0]), clusters)

    clusters, _ = run_clustering_experiment(
        TimeSeriesKMedoids(
            n_clusters=5, max_iter=50, metric="euclidean", random_state=rng
        ),
        X_train,
        X_test,
    )

    assert np.array_equal(np.array([1, 3, 3, 4, 4, 3, 4, 3, 2, 4]), clusters)

    clusters, _ = run_clustering_experiment(
        TimeSeriesKMedoids(
            n_clusters=5,
            max_iter=50,
            init_algorithm="random",
            metric="euclidean",
            random_state=rng,
        ),
        X_train,
        X_test,
    )

    assert np.array_equal(np.array([2, 1, 1, 3, 3, 0, 1, 0, 1, 3]), clusters)
