# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering._k_means import TimeSeriesKMeans
from sktime.clustering.tests._clustering_tests import (
    generate_univaritate_series,
    run_clustering_experiment,
)


def test_k_means():
    rng = np.random.RandomState(1)
    X_train = generate_univaritate_series(n=100, size=5, rng=rng, dtype=np.double)
    X_test = generate_univaritate_series(
        n=10, size=5, rng=np.random.RandomState(2), dtype=np.double
    )

    clusters, _ = run_clustering_experiment(
        TimeSeriesKMeans(
            n_clusters=5,
            max_iter=50,
            metric="euclidean",
            averaging_algorithm="mean",
            init_algorithm="forgy",
            random_state=rng,
        ),
        X_train,
        X_test,
    )
    assert np.array_equal(np.array([3, 1, 0, 2, 0, 1, 1, 1, 1, 0]), clusters)

    # Bug with dtw as metric that is only works if the array is type double
    clusters, _ = run_clustering_experiment(
        TimeSeriesKMeans(
            n_clusters=5,
            max_iter=50,
            metric="dtw",
            averaging_algorithm="mean",
            random_state=rng,
        ),
        X_train,
        X_test,
    )
    assert np.array_equal(np.array([2, 3, 3, 4, 4, 3, 3, 3, 2, 3]), clusters)

    clusters, _ = run_clustering_experiment(
        TimeSeriesKMeans(
            n_clusters=5,
            max_iter=5,
            averaging_algorithm="dba",
            averaging_algorithm_iterations=2,
            random_state=rng,
        ),
        X_train,
        X_test,
    )

    # Need to add seeding to dba so this works
    assert clusters

    clusters, _ = run_clustering_experiment(
        TimeSeriesKMeans(
            n_clusters=5,
            max_iter=5,
            init_algorithm="random",
            averaging_algorithm="dba",
            averaging_algorithm_iterations=2,
            random_state=rng,
        ),
        X_train,
        X_test,
    )

    assert np.array_equal(np.array([1, 4, 3, 3, 4, 4, 1, 1, 1, 4]), clusters)
