#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = [
    "TEST_YS",
    "TEST_SPS",
    "TEST_ALPHAS",
    "TEST_FHS",
    "TEST_STEP_LENGTHS",
    "TEST_INS_FHS",
    "TEST_OOS_FHS",
    "TEST_WINDOW_LENGTHS",
    "VALID_INDEX_FH_COMBINATIONS",
    "INDEX_TYPE_LOOKUP",
    "TEST_RANDOM_SEEDS",
    "TEST_N_ITERS",
]

import numpy as np
import pandas as pd

from sktime.utils._testing.series import _make_series

# We here define the parameter values for unit testing.
TEST_WINDOW_LENGTHS = [1, 5]
TEST_STEP_LENGTHS = [1, 5]
TEST_OOS_FHS = [1, np.array([2, 5])]  # out-of-sample
TEST_INS_FHS = [
    -3,  # single in-sample
    np.array([-2, -5]),  # multiple in-sample
    0,  # last training point
    np.array([-3, 2]),  # mixed in-sample and out-of-sample
]
TEST_FHS = TEST_OOS_FHS + TEST_INS_FHS
TEST_SPS = [3, 12]
TEST_ALPHAS = [0.05, 0.1]
TEST_YS = [
    _make_series(all_positive=True),
]
TEST_RANDOM_SEEDS = [1, 42]
TEST_N_ITERS = [1, 4]

# We currently support the following combinations of index and forecasting horizon types
VALID_INDEX_FH_COMBINATIONS = [
    # index type, fh type, is_relative
    ("int", "int", True),
    ("int", "int", False),
    ("range", "int", True),
    ("range", "int", False),
    ("period", "int", True),
    ("period", "period", False),
    ("datetime", "int", True),
    ("datetime", "datetime", False),
]

INDEX_TYPE_LOOKUP = {
    "int": pd.Int64Index,
    "range": pd.RangeIndex,
    "datetime": pd.DatetimeIndex,
    "period": pd.PeriodIndex,
}
