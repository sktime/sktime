# -*- coding: utf-8 -*-
from sktime.clustering.hierarchical._time_series_feature_agglomeration import (
    TimeSeriesFeatureAgglomerative,
)
from sktime.clustering.types import (
    Data_Frame,
)


def test_time_series_feat_agglo_clust(df_x: Data_Frame, df_y: Data_Frame):
    m = TimeSeriesFeatureAgglomerative()
    m.fit(df_x)
