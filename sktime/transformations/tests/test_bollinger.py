"""Tests for Bollinger."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ishanpai"]
__all__ = []

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.bollinger import Bollinger


@pytest.mark.skipif(
    not run_test_for_class(Bollinger),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("window", [2, 12])
@pytest.mark.parametrize("k", [1, 5])
def test_bollinger_against_raw_implementation(window, k):
    y = load_airline()
    t = Bollinger(window=window, k=k)
    y_t = t.fit_transform(y)

    ma = y.rolling(window=window).mean()
    std = y.rolling(window=window).std()

    np.testing.assert_array_equal(y_t["moving_average"].values, ma.values)
    np.testing.assert_array_equal(y_t["upper"].values, (ma + k * std).values)
    np.testing.assert_array_equal(y_t["lower"].values, (ma - k * std).values)


@pytest.mark.skipif(
    not run_test_for_class(Bollinger),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(("window", "k"), [(1, 1), (12, 0), (1, 0)])
def test_bollinger_input_error(window, k):
    with pytest.raises(ValueError):
        Bollinger(window=window, k=k)
