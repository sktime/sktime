# -*- coding: utf-8 -*-
import os
import sktime

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.clustering._time_series_k_medoids import TimeSeriesKMedoids
from sktime.clustering.evaluation._plot_clustering import plot_cluster_algorithm

DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")
X, Y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts")
)

X_test, Y_test = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TEST.ts")
)

plot_results = False


def __run_test(model: TimeSeriesKMedoids):
    model.fit(X)
    if plot_results:
        plot_cluster_algorithm(model, X_test)


def test_k_medoids():
    __run_test(TimeSeriesKMedoids(n_clusters=5, max_iter=50))
    __run_test(TimeSeriesKMedoids(n_clusters=5, max_iter=50, metric="euclidean"))
