# -*- coding: utf-8 -*-
from sktime.clustering.partition_based._time_series_affinity_propagation import (
    TimeSeriesAffinityPropagation,
)
from sktime.clustering.utils import (
    Numpy_Array,
    convert_df_to_sklearn_format,
    Data_Frame,
)


def test_time_series_affinity_propagation(df_x: Data_Frame, df_y: Data_Frame):
    sklearn_train_data: Numpy_Array = convert_df_to_sklearn_format(df_x)
    ap = TimeSeriesAffinityPropagation(
        damping=0.5,
        max_iter=200,
        copy=True,
        preference=None,
        affinity="euclidean",
        verbose=False,
        random_state=0,
    )
    ap.fit_predict(sklearn_train_data)
