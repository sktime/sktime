# -*- coding: utf-8 -*-
import os
import sktime

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.clustering._time_series_k_means import TimeSeiresKMeans


DATA_PATH = os.path.join(os.path.dirname(
    sktime.__file__), "datasets/data")
X, Y = \
    load_from_tsfile_to_dataframe(os.path.join(
        DATA_PATH, "ArrowHead/ArrowHead_TRAIN.ts"))


def test_k_means():
    # print(DATA_PATH)
    model = TimeSeiresKMeans()
    model.fit(X)
    # print("done")
