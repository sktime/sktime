"""Tests for Bollinger."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ishanpai"]
__all__ = []

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.transformations.series.bollinger import Bollinger


@pytest.mark.parametrize("window", [2, 12])
@pytest.mark.parametrize("k", [1, 5])
def test_bollinger_against_raw_implementation(window, k):
    y = load_airline()
    t = Bollinger(window=window, k=k)
    y_t = t.fit_transform(y)

    ma = y.rolling(window=window).mean()
    std = y.rolling(window=window).std()

    np.testing.assert_array_equal(
        y_t.xs("moving_average", axis=1, level=0).values, ma.values
    )
    np.testing.assert_array_equal(
        y_t.xs("upper", axis=1, level=0).values, (ma + k * std).values
    )
    np.testing.assert_array_equal(
        y_t.xs("lower", axis=1, level=0).values, (ma - k * std).values
    )


def test_bollinger_window_input_error():
    y = load_airline()
    t = Bollinger(window=1, k=1)
    try:
        t.fit_transform(y)
    except AssertionError("window must be greater than 1, passed 1"):
        return True


def test_bollinger_k_input_error():
    y = load_airline()
    t = Bollinger(window=12, k=0)
    try:
        t.fit_transform(y)
    except AssertionError("k must be positive, passed 0"):
        return True
