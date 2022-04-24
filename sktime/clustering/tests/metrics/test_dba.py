# -*- coding: utf-8 -*-
"""Tests for DBA."""

import time

from tslearn.barycenters import (
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_petitjean,
)

from sktime.clustering.metrics.averaging import dba
from sktime.distances.tests._utils import create_test_distance_numpy


def _tslearn_time(X_train, callable):
    X = X_train.copy()
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    start = time.time()
    callable(X, max_iter=10)
    total = time.time() - start
    return total


def _timing_exper(X_train):

    start = time.time()
    dba(X_train)
    sktime_dba = time.time() - start

    tslearn_paj = _tslearn_time(X_train, dtw_barycenter_averaging_petitjean)
    tslearn_reg = _tslearn_time(X_train, dtw_barycenter_averaging)

    return sktime_dba, tslearn_paj, tslearn_reg


def test_dba():
    """Test dba."""
    X_train = create_test_distance_numpy(100, 100, 100, random_state=2)

    # time_path(X_train)

    result = _timing_exper(X_train)
    print("\n")  # noqa: T001
    print(f"sktime: {result[0]}")  # noqa: T001
    print(f"tslearn paj: {result[1]}")  # noqa: T001
    print(f"tslearn reg: {result[2]}")  # noqa: T001


def test_plot():
    """Plot dba."""
    import matplotlib.pyplot as plt
    import numpy
    from tslearn.datasets import CachedDatasets

    # fetch the example data set
    numpy.random.seed(0)
    X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")
    # tslearn_X = X_train[y_train == 2]

    tslearn_X = X_train
    length_of_sequence = tslearn_X.shape[1]

    sktime_X = tslearn_X.copy()
    sktime_X = sktime_X.reshape(
        (sktime_X.shape[0], sktime_X.shape[2], sktime_X.shape[1])
    )

    def plot_helper(barycenter):
        # plot all points of the data set
        for series in sktime_X:
            plt.plot(series.ravel(), "k-", alpha=0.2)
        # plot the given barycenter of them
        plt.plot(barycenter.ravel(), "r-", linewidth=2)

    # plot the four variants with the same number of iterations and a tolerance of
    # 1e-3 where applicable
    ax1 = plt.subplot()

    plt.subplot(4, 1, 1, sharex=ax1)
    plt.title("Sktime DBA (using dtw)")
    plot_helper(dba(sktime_X, distance_metric="dtw", medoids_distance_metric="dtw"))

    plt.subplot(4, 1, 2, sharex=ax1)
    plt.title("Sktime DBA (using wdtw)")
    plot_helper(dba(sktime_X, distance_metric="wdtw", medoids_distance_metric="wdtw"))

    plt.subplot(4, 1, 3, sharex=ax1)
    plt.title("Sktime DBA (using lcss)")
    plot_helper(dba(sktime_X, distance_metric="lcss", medoids_distance_metric="lcss"))

    plt.subplot(4, 1, 4, sharex=ax1)
    plt.title("Sktime DBA (using msm)")
    plot_helper(dba(sktime_X, distance_metric="msm"))

    # test = medoids(sktime_X)
    # plt.subplot(4, 1, 2, sharex=ax1)
    # plt.title("Medoids algo")
    # plot_helper(medoids(sktime_X.copy(0)))
    #
    # plt.subplot(4, 1, 4, sharex=ax1)
    # plt.title("Tslearn DBA")
    # plot_helper(dtw_barycenter_averaging_petitjean(tslearn_X))

    # clip the axes for better readability
    ax1.set_xlim([0, length_of_sequence])

    # show the plot(s)
    plt.tight_layout()
    plt.show()
