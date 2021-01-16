# -*- coding: utf-8 -*-
from sktime.clustering.density_based._time_series_mean_shift import TimeSeriesMeanShift
from sktime.clustering.types import (
    Data_Frame,
)


def test_time_series_mean_shift(df_x: Data_Frame, df_y: Data_Frame):
    km = TimeSeriesMeanShift()
    km.fit_predict(df_x)
