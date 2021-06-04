# -*- coding: utf-8 -*-
import os
import sktime

from sktime.clustering.partitioning._averaging_metrics import (
    BarycenterAveraging,
    MeanAveraging,
)
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_processing import from_nested_to_2d_array

# import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(sktime.__file__), "datasets/data")
X, Y = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts")
)

X_test, Y_test = load_from_tsfile_to_dataframe(
    os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TEST.ts")
)


def test_barycenter_averaging():
    sub_section = X.sample(n=10)
    values = from_nested_to_2d_array(sub_section, return_numpy=True)

    BCA = BarycenterAveraging(values)
    BCA.average()
    # average_series = BCA.average()

    # plt.figure()
    # plt.plot(range(0, len(average_series)), average_series)
    # plt.show()


def test_mean_averaging():
    sub_section = X.sample(n=10)
    values = from_nested_to_2d_array(sub_section, return_numpy=True)

    mean = MeanAveraging(values)
    mean.average()

    # average_series = mean.average()
    # plt.figure()
    # plt.plot(range(0, len(average_series)), average_series)
    # plt.show()
