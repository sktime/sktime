#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Ansgar Asseburg"]
__all__ = [
    "TEST_YS",
    "TEST_YS_ZERO",
]

import pandas as pd
from sktime.utils._testing.series import _make_series


RANDOM_SEED = 42
TEST_YS = [
    [_make_series(n_timepoints=50, random_state=RANDOM_SEED),
     _make_series(n_timepoints=60, random_state=RANDOM_SEED)],
    [_make_series(n_timepoints=80, random_state=RANDOM_SEED),
     _make_series(n_timepoints=50, random_state=RANDOM_SEED)]
]


TEST_YS_ZERO = [pd.Series([0.0, 0.0, 0.0])]
