# -*- coding: utf-8 -*-
from sktime.clustering.density_based._time_series_dbscan import TimeSeriesDBSCAN
from sktime.clustering.types import (
    Data_Frame,
)


def test_time_series_dbscan(df_x: Data_Frame, df_y: Data_Frame):
    m = TimeSeriesDBSCAN()
    m.fit_predict(df_x)
