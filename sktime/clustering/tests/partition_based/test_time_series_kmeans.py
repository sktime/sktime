# -*- coding: utf-8 -*-
from sktime.clustering.partition_based._time_series_kmeans import TimeSeriesKMeans
from sktime.clustering.utils import (
    convert_df_to_sklearn_format,
    Data_Frame,
    SkLearn_Data,
)


def test_time_series_kmeans():
    TimeSeriesKMeans()


def test_create_sklearn_k_means(df_x: Data_Frame, df_y: Data_Frame):
    sklearn_train_data: SkLearn_Data = convert_df_to_sklearn_format(df_x)
    km = TimeSeriesKMeans(
        n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0
    )
    km.fit(sklearn_train_data)
