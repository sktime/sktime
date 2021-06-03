# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from sktime.clustering.base.base_types import Data_Frame
from sktime.clustering.base.base import BaseCluster

__author__ = "Christopher Holder"


def __plot_test(clusters, center):
    for cluster in clusters["dim_0"]:
        cluster.plot(color="b")

    center.iloc[0].plot(color="r")
    plt.show()


def plot_cluster_algorithm(model: BaseCluster, predict_series: Data_Frame):
    indexes = model.predict(predict_series)
    centers = model.get_centers()
    for i in range(len(indexes)):
        series = predict_series.iloc[indexes[i]]
        center = centers.iloc[i]
        __plot_test(series, center)
