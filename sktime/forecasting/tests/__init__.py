#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd

# default parameter testing grid
DEFAULT_WINDOW_LENGTHS = [1, 5]
DEFAULT_STEP_LENGTHS = [1, 5]
DEFAULT_FHS = [1, np.array([2, 5])]
DEFAULT_INSAMPLE_FHS = [-3, np.array([-2, -5]), 0, np.array([-3, 2])]
DEFAULT_SPS = [3, 7, 12]


def make_forecasting_problem(n_timepoints=50):
    n_train = n_timepoints - 20
    s = pd.Series(np.arange(n_timepoints))
    y_train = s.iloc[:n_train]
    y_test = s.iloc[n_train:]
    return y_train, y_test
