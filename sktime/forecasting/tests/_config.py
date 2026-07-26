#!/usr/bin/env python3 -u

__author__ = ["mloning"]
__all__ = [
    "TEST_YS",
    "TEST_SPS",
    "TEST_ALPHAS",
    "TEST_FHS",
    "TEST_STEP_LENGTHS_INT",
    "TEST_STEP_LENGTHS",
    "TEST_INS_FHS",
    "TEST_OOS_FHS",
    "TEST_WINDOW_LENGTHS_INT",
    "TEST_WINDOW_LENGTHS",
    "TEST_INITIAL_WINDOW_INT",
    "TEST_INITIAL_WINDOW",
    "VALID_INDEX_FH_COMBINATIONS",
    "INDEX_TYPE_LOOKUP",
    "TEST_RANDOM_SEEDS",
    "TEST_N_ITERS",
    "TEST_FHS_TIMEDELTA",
    "TEST_CUTOFFS",
]

import numpy as np
import pandas as pd

from sktime.utils._testing.series import _make_series

# We here define the parameter values for unit testing.
TEST_CUTOFFS_INT_LIST = [[21, 22], [3, 7, 10]]
TEST_CUTOFFS_INT_ARR = [np.array([21, 22]), np.array([3, 7, 10])]
# The following timestamps correspond
# to the above integers for `_make_series(all_positive=True)`
TEST_CUTOFFS_TIMESTAMP = [
    pd.to_datetime(["2000-01-22", "2000-01-23"]),
    pd.to_datetime(["2000-01-04", "2000-01-08", "2000-01-11"]),
]
TEST_CUTOFFS = [*TEST_CUTOFFS_INT_LIST, *TEST_CUTOFFS_INT_ARR, *TEST_CUTOFFS_TIMESTAMP]

TEST_WINDOW_LENGTHS_INT = [1, 5]
TEST_WINDOW_LENGTHS_TIMEDELTA = [pd.Timedelta(1, unit="D"), pd.Timedelta(5, unit="D")]
TEST_WINDOW_LENGTHS_DATEOFFSET = [pd.offsets.Day(1), pd.offsets.Day(5)]
TEST_WINDOW_LENGTHS = [
    *TEST_WINDOW_LENGTHS_INT,
    *TEST_WINDOW_LENGTHS_TIMEDELTA,
    *TEST_WINDOW_LENGTHS_DATEOFFSET,
]

TEST_INITIAL_WINDOW_INT = [7, 10]
TEST_INITIAL_WINDOW_TIMEDELTA = [pd.Timedelta(7, unit="D"), pd.Timedelta(10, unit="D")]
TEST_INITIAL_WINDOW_DATEOFFSET = [pd.offsets.Day(7), pd.offsets.Day(10)]
TEST_INITIAL_WINDOW = [
    *TEST_INITIAL_WINDOW_INT,
    *TEST_INITIAL_WINDOW_TIMEDELTA,
    *TEST_INITIAL_WINDOW_DATEOFFSET,
]

TEST_STEP_LENGTHS_INT = [1, 5]
TEST_STEP_LENGTHS_TIMEDELTA = [pd.Timedelta(1, unit="D"), pd.Timedelta(5, unit="D")]
TEST_STEP_LENGTHS_DATEOFFSET = [pd.offsets.Day(1), pd.offsets.Day(5)]
TEST_STEP_LENGTHS = [
    *TEST_STEP_LENGTHS_INT,
    *TEST_STEP_LENGTHS_TIMEDELTA,
    *TEST_STEP_LENGTHS_DATEOFFSET,
]

TEST_OOS_FHS = [1, np.array([2, 5], dtype="int64")]  # out-of-sample
TEST_INS_FHS = [
    -3,  # single in-sample
    np.array([-2, -5], dtype="int64"),  # multiple in-sample
    0,  # last training point
    np.array([-3, 2], dtype="int64"),  # mixed in-sample and out-of-sample
]
TEST_FHS = [*TEST_OOS_FHS, *TEST_INS_FHS]

TEST_OOS_FHS_TIMEDELTA = [
    [pd.Timedelta(1, unit="D")],
    [pd.Timedelta(2, unit="D"), pd.Timedelta(5, unit="D")],
]  # out-of-sample
TEST_INS_FHS_TIMEDELTA = [
    pd.Timedelta(-3, unit="D"),  # single in-sample
    [pd.Timedelta(-2, unit="D"), pd.Timedelta(-5, unit="D")],  # multiple in-sample
    pd.Timedelta(0, unit="D"),  # last training point
    [
        pd.Timedelta(-3, unit="D"),
        pd.Timedelta(2, unit="D"),
    ],  # mixed in-sample and out-of-sample
]
TEST_FHS_TIMEDELTA = [*TEST_OOS_FHS_TIMEDELTA, *TEST_INS_FHS_TIMEDELTA]

TEST_SPS = [3, 12]
TEST_ALPHAS = [0.05, 0.1, [0.25, 0.75]]
TEST_YS = [_make_series(all_positive=True)]
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
    ("datetime", "timedelta", True),
]

INDEX_TYPE_LOOKUP = {
    "int": pd.Index,
    "range": pd.RangeIndex,
    "datetime": pd.DatetimeIndex,
    "period": pd.PeriodIndex,
    "timedelta": pd.TimedeltaIndex,
}
