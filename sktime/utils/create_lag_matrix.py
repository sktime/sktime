#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)


__author__ = ["Sebastiaan Koel"]
__all__ = ["create_lag_matrix"]

import numpy as np
import pandas as pd


def create_lag_matrix(x, window_length):
    is_pandas = 1
    if isinstance(x, pd.DataFrame):
        cols = list(x.columns)
    elif isinstance(x, pd.Series):
        cols = [x.name]
    else:
        is_pandas = 0

    # input checks
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # time series tabularisation
    n_timepoints, n_variables = x.shape
    lm = np.zeros((n_timepoints + window_length, n_variables * (window_length)))
    for k in range(window_length):
        a = window_length - k
        b = n_timepoints + window_length - k
        # get correct index for placing the columns
        idx = [
            (i * window_length) + (window_length - k - 1) for i in range(n_variables)
        ]
        lm[a:b, idx] = x

    # separate y, X and truncate
    y = x[window_length:, 0]
    X = lm[window_length:-window_length]

    if is_pandas:
        y = pd.Series(y, name=cols[0])
        cols = [f"{j}_min_{i}" for j in cols for i in range(1, window_length + 1)]
        X = pd.DataFrame(X, columns=cols)

    return y, X
