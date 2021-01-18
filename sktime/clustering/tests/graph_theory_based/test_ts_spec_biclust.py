# -*- coding: utf-8 -*-
from sktime.clustering.graph_theory_based._time_series_spectral_biclustering import (
    TimeSeriesSpectralBiclustering,
)
from sktime.clustering.utils import (
    Numpy_Array,
    convert_df_to_sklearn_format,
    Data_Frame,
)


def test_time_series_spectral_biclustering(df_x: Data_Frame, df_y: Data_Frame):
    sklearn_train_data: Numpy_Array = convert_df_to_sklearn_format(df_x)
    m = TimeSeriesSpectralBiclustering()
    m.fit(sklearn_train_data)
