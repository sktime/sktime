# -*- coding: utf-8 -*-
from sktime.clustering.hierarchical._time_series_birch import TimeSeriesBirch
from sktime.clustering.types import (
    Data_Frame,
)


def test_time_series_birch(df_x: Data_Frame, df_y: Data_Frame):
    m = TimeSeriesBirch()
    m.fit_predict(df_x)
