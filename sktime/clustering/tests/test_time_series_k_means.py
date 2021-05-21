# -*- coding: utf-8 -*-
import os


import sktime

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.clustering._time_series_k_means import TimeSeriesKMeans


DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")
X, Y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts")
)

X_test, Y_test = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TEST.ts")
)


# import matplotlib.pyplot as plt
# def plot_test(clusters, center):
#     for cluster in clusters["dim_0"]:
#         cluster.plot(color="b")
#
#     center.iloc[0].plot(color="r")
#     plt.show()


def test_k_means():
    model = TimeSeriesKMeans(n_clusters=5, max_iter=300)
    model.fit(X)
    indexes = model.predict(X_test)
    centers = model.get_centers()
    # for i in range(len(indexes)):
    #     series = X_test.iloc[indexes[i]]
    #     center = centers.iloc[i]
    #     plot_test(series, center)

    # This is just temp to get it past initial linting errors
    if indexes is not None and centers is not None:
        return True
    return False
