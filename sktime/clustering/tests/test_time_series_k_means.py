# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sktime

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.clustering._time_series_k_means import TimeSeiresKMeans


DATA_PATH = os.path.join(os.path.dirname(
    sktime.__file__), "datasets/data")
X, Y = load_from_tsfile_to_dataframe(os.path.join(
        DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts"))

X_test, Y_test = load_from_tsfile_to_dataframe(os.path.join(
        DATA_PATH, "ArrowHead/ArrowHead_TEST.ts"))

def plot_test(clusters, center):
    for cluster in clusters:
        cluster.plot(color='b')

    center.plot(color='r')
    plt.show()



def test_k_means():
    print("\n==================")
    model = TimeSeiresKMeans(n_clusters=5)
    model.fit(X)
    clusters = model.predict(X_test)
    centers = model.get_centroids()

    for i in range(len(clusters)):
        plot_test(clusters[i], centers[i])

    # print("done")
