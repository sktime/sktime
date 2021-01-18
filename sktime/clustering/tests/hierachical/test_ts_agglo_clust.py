# -*- coding: utf-8 -*-
from sktime.clustering.hierarchical._time_series_agglomerative_clustering import (
    TimeSeriesAgglomerativeClustering,
)
from sktime.clustering.types import (
    Data_Frame,
)


def test_time_series_agglo_clust(df_x: Data_Frame, df_y: Data_Frame):
    m = TimeSeriesAgglomerativeClustering()
    m.fit_predict(df_x)
