# -*- coding: utf-8 -*-
from sktime.clustering.partition_based._time_series_affinity_propagation import (
    TimeSeriesAffinityPropagation,
)
from sktime.clustering.types import (
    Data_Frame,
)


def test_time_series_affinity_propagation(df_x: Data_Frame, df_y: Data_Frame):
    ap = TimeSeriesAffinityPropagation(
        damping=0.5,
        max_iter=200,
        copy=True,
        preference=None,
        affinity="euclidean",
        verbose=True,
        random_state=0,
    )
    ap.fit_predict(df_x)
