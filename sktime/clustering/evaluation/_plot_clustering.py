# -*- coding: utf-8 -*-
"""Cluster plotting tools"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["plot_cluster_algorithm"]

import pandas as pd
from sktime.clustering.base._typing import NumpyOrDF
from sktime.clustering.base.base import BaseClusterer
from sktime.clustering.partitioning._lloyds_partitioning import (
    TimeSeriesLloydsPartitioning,
)
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.utils.validation._dependencies import _check_soft_dependencies


def _plot(cluster_values, center, axes):
    for cluster_series in cluster_values:
        axes.plot(cluster_series, color="b")

    axes.plot(center, color="r")


def plot_cluster_algorithm(model: BaseClusterer, predict_series: NumpyOrDF, k: int):
    """
    Method that is used to plot a clustering algorithms output

    Parameters
    ----------
    model: BaseClusterer
        Clustering model to plot

    predict_series: Numpy or Dataframe
        The series to predict the values for

    k: int
        Number of centers
    """
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if isinstance(predict_series, pd.DataFrame):
        predict_series = from_nested_to_2d_array(predict_series, return_numpy=True)
    plt.figure(figsize=(5, 10))
    plt.rcParams["figure.dpi"] = 100
    indexes = model.predict(predict_series)
    centers = model.get_centers()

    series_values = TimeSeriesLloydsPartitioning.get_cluster_values(
        indexes, predict_series, k
    )
    fig, axes = plt.subplots(nrows=k, ncols=1)
    for i in range(k):
        _plot(series_values[i], centers[i], axes[i])

    blue_patch = mpatches.Patch(color="blue", label="Series that belong to the cluster")
    red_patch = mpatches.Patch(color="red", label="Cluster centers")
    plt.legend(
        handles=[red_patch, blue_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.tight_layout()
    plt.show()
