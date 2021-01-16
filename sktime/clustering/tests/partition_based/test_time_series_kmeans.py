# -*- coding: utf-8 -*-
from sktime.clustering.partition_based._time_series_kmeans import TimeSeriesKMeans
from sktime.clustering.types import (
    Data_Frame,
)


def test_time_series_kmeans(df_x: Data_Frame, df_y: Data_Frame):
    km = TimeSeriesKMeans(
        n_clusters=3,
        init="random",
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=0,
        verbose=1,
    )
    km.fit_predict(df_x)
