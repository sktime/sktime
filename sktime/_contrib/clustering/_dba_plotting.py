# -*- coding: utf-8 -*-
"""Plotting examples for DBA."""

from sktime.clustering.metrics.averaging import dba
from sktime.datasets import load_arrow_head
from sktime.datatypes import convert_to


def plot_dba_example():
    """Plot dba."""
    import matplotlib.pyplot as plt

    X_train, y_train = load_arrow_head(split="train")
    X_train = convert_to(X_train, "numpy3D")

    def plot_helper(barycenter):
        for series in X_train:
            plt.plot(series.ravel(), "k-", alpha=0.2)
        plt.plot(barycenter.ravel(), "r-", linewidth=2)

    ax1 = plt.subplot()

    plt.subplot(4, 1, 1, sharex=ax1)
    plt.title("Sktime DBA (using dtw)")
    plot_helper(dba(X_train, distance_metric="dtw", medoids_distance_metric="dtw"))

    plt.subplot(4, 1, 2, sharex=ax1)
    plt.title("Sktime DBA (using wdtw)")
    plot_helper(dba(X_train, distance_metric="wdtw", medoids_distance_metric="wdtw"))

    plt.subplot(4, 1, 3, sharex=ax1)
    plt.title("Sktime DBA (using lcss)")
    plot_helper(dba(X_train, distance_metric="lcss", medoids_distance_metric="lcss"))

    plt.subplot(4, 1, 4, sharex=ax1)
    plt.title("Sktime DBA (using msm)")
    plot_helper(dba(X_train, distance_metric="msm"))

    ax1.set_xlim([0, X_train.shape[2]])

    # show the plot(s)
    plt.tight_layout()
    plt.show()
