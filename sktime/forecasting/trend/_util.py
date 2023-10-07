# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements util functions for trend forecasters."""

__author__ = ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"]

import pandas as pd


def _get_X_numpy_int_from_pandas(x):
    """Convert pandas index to an sklearn compatible X, 2D np.ndarray, int type."""
    if isinstance(x, (pd.DatetimeIndex)):
        x = x.astype("int64") / 864e11
    else:
        x = x.astype("int64")
    return x.to_numpy().reshape(-1, 1)
