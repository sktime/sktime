#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = ["test_plot_series_uniform_treatment_of_int64_range_index_types"]


import pandas as pd
import numpy as np
from sktime.utils.plotting import plot_series


def test_plot_series_uniform_treatment_of_int64_range_index_types():
    # We test that int64 and range indices are treated uniformly and do not raise an
    # error of inconsistent index types
    y1 = pd.Series(np.arange(10))
    y2 = pd.Series(np.random.normal(size=10))
    y1.index = pd.Int64Index(y1.index)
    y2.index = pd.RangeIndex(y2.index)
    plot_series(y1, y2)
