# -*- coding: utf-8 -*-
from sktime.clustering.density_based._time_series_mean_shift import TimeSeriesMeanShift
from sktime.clustering.utils import (
    convert_df_to_sklearn_format,
    Data_Frame,
    SkLearn_Data,
)


def test_time_series_mean_shift(df_x: Data_Frame, df_y: Data_Frame):
    sklearn_train_data: SkLearn_Data = convert_df_to_sklearn_format(df_x)
    km = TimeSeriesMeanShift()
    km.fit_predict(sklearn_train_data)
