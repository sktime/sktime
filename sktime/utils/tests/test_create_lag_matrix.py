#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)


__author__ = ["Sebastiaan Koel"]
__all__ = []

import numpy as np
import pandas as pd
from sktime.utils.create_lag_matrix import create_lag_matrix


def test_create_lag_matrix():
    y_len = 10
    window_length = 2
    np.random.seed(0)
    df = pd.DataFrame(
        np.random.randint(0, y_len, size=(y_len, 4)), columns=list("ABCD")
    )

    y, X = create_lag_matrix(df, window_length=window_length)

    assert len(y) == X.shape[0]
    assert len(y) == (y_len - window_length)
    assert X.shape[1] == df.shape[1] * window_length
    np.testing.assert_array_equal(df.iloc[window_length:, 0], y)

    for col in df.columns:
        for lag in range(1, window_length + 1):
            rows = list(range(window_length - lag, y_len - lag))
            np.testing.assert_array_equal(
                df.loc[rows, col], X.loc[:, f"{col}_min_{lag}"]
            )
