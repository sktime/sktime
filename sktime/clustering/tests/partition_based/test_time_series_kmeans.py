# -*- coding: utf-8 -*-
from sktime.clustering.partition_based._time_series_kmeans import TimeSeriesKMeans
from sktime.clustering.utils import (
    Numpy_Array,
    convert_df_to_sklearn_format,
    Data_Frame,
)


def test_time_series_kmeans(df_x: Data_Frame, df_y: Data_Frame):
    sklearn_train_data: Numpy_Array = convert_df_to_sklearn_format(df_x)
    km = TimeSeriesKMeans(
        n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0
    )
    km.fit_predict(sklearn_train_data)
