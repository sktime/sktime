# -*- coding: utf-8 -*-

import numpy as np

from sktime.clustering.metrics.averaging import dba
from sktime.clustering.metrics.medoids import medoids
from sktime.datasets import load_acsf1
from sktime.datatypes import convert_to
from sktime.distances.tests._utils import create_test_distance_numpy
from tslearn.barycenters import dtw_barycenter_averaging_petitjean


def test_dba():
    """Test medoids."""
    X_train, y_train = load_acsf1(split="train")
    X_test, y_test = load_acsf1(split="test")
    # X_train = create_test_distance_numpy(10, 4, 3, random_state=2)

    X_train = convert_to(X_train, "numpy3D")

    X_train = X_train[:5]

    test_dba = dba(X_train)
    test_medoids = medoids(X_train)
    joe = ""


def test_tslearn_ploat():
    import numpy
    import matplotlib.pyplot as plt

    from tslearn.barycenters import dtw_barycenter_averaging
    from tslearn.datasets import CachedDatasets

    # fetch the example data set
    numpy.random.seed(0)
    X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")
    # tslearn_X = X_train[y_train == 2]

    tslearn_X = X_train
    length_of_sequence = tslearn_X.shape[1]

    sktime_X = tslearn_X.copy()
    sktime_X = sktime_X.reshape((sktime_X.shape[0], sktime_X.shape[2], sktime_X.shape[1]))

    def plot_helper(barycenter):
        # plot all points of the data set
        for series in sktime_X:
            plt.plot(series.ravel(), "k-", alpha=.2)
        # plot the given barycenter of them
        plt.plot(barycenter.ravel(), "r-", linewidth=2)

    # plot the four variants with the same number of iterations and a tolerance of
    # 1e-3 where applicable
    ax1 = plt.subplot()

    plt.subplot(4, 1, 1, sharex=ax1)
    plt.title("Sktime DBA (using dtw)")
    plot_helper(dba(sktime_X, distance_metric='dtw'))

    plt.subplot(4, 1, 2, sharex=ax1)
    plt.title("Sktime DBA (using wdtw)")
    plot_helper(dba(sktime_X, distance_metric='wdtw'))
    #
    plt.subplot(4, 1, 3, sharex=ax1)
    plt.title("Sktime DBA (using lcss)")
    plot_helper(dba(sktime_X, distance_metric='lcss'))

    plt.subplot(4, 1, 4, sharex=ax1)
    plt.title("Sktime DBA (using msm)")
    plot_helper(dba(sktime_X, distance_metric='msm'))

    # plt.subplot(4, 1, 4, sharex=ax1)
    # plt.title("Tslearn DBA")
    # plot_helper(dtw_barycenter_averaging_petitjean(tslearn_X))

    # clip the axes for better readability
    ax1.set_xlim([0, length_of_sequence])

    # show the plot(s)
    plt.tight_layout()
    plt.show()
