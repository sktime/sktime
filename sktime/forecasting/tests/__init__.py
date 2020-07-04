#!/usr/bin/env python3 -u
# coding: utf-8
<<<<<<< HEAD
=======
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

__author__ = ["Markus Löning"]

import numpy as np
<<<<<<< HEAD
import pandas as pd
from sktime.utils.testing.forecasting import generate_time_series
=======
from sktime.utils._testing.forecasting import generate_time_series
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed

# default parameter testing grid
TEST_WINDOW_LENGTHS = [1, 5]
TEST_STEP_LENGTHS = [1, 5]
TEST_OOS_FHS = [1, np.array([2, 5])]  # out-of-sample
TEST_INS_FHS = [
    -3,  # single in-sample
    np.array([-2, -5]),  # multiple in-sample
    0,  # last training point
    np.array([-3, 2])  # mixed in-sample and out-of-sample
]
TEST_FHS = TEST_OOS_FHS + TEST_INS_FHS
TEST_SPS = [3, 7, 12]
TEST_ALPHAS = [0.05, 0.1]

n_timepoints = 50
TEST_YS = [
<<<<<<< HEAD
    generate_time_series(positive=True),  # zero-based index
    generate_time_series(positive=True, non_zero_index=True),  # non-zero-based index
=======
    # zero-based index
    generate_time_series(positive=True),
    # non-zero-based index, raises warnings in statsmodels
    # generate_time_series(positive=True, non_zero_index=True),
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
]
