# -*- coding: utf-8 -*-
import numpy as np

from sktime.clustering._time_series_k_means import TimeSeriesKMeans

# import os
# import sktime
# from sktime.clustering.evaluation._plot_clustering import plot_cluster_algorithm
# from sktime.utils.data_io import load_from_tsfile_to_dataframe

# DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")
# X, Y = load_from_tsfile_to_dataframe(
#     os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts")
# )
#
# X_test, Y_test = load_from_tsfile_to_dataframe(
#     os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TEST.ts")
# )


def __run_test(model: TimeSeriesKMeans):
    n, sz = 100, 10
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz)
    rng = np.random.RandomState(1)
    X_test = rng.randn(n, sz)

    model.fit(X)
    model.predict(X_test)
    # import matplotlib.pyplot as plt
    # from sktime.clustering.evaluation._plot_clustering import plot_cluster_algorithm
    #
    # plot_cluster_algorithm(model, X_test, model.n_clusters, plt)


def test_k_means():
    __run_test(
        TimeSeriesKMeans(
            n_clusters=5,
            max_iter=5,
            averaging_algorithm="dba",
            averaging_algorithm_iterations=2,
        )
    )
    __run_test(
        TimeSeriesKMeans(
            n_clusters=5,
            max_iter=50,
            metric="euclidean",
            averaging_algorithm="mean",
            init_algorithm="forgy",
        )
    )
    __run_test(TimeSeriesKMeans(n_clusters=5, max_iter=50, averaging_algorithm="mean"))
