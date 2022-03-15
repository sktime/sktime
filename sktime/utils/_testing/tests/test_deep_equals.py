# -*- coding: utf-8 -*-
"""Tests for deep_equals utility."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.utils._testing.deep_equals import deep_equals

# examples used for comparison below
EXAMPLES = [
    42,
    [],
    ((((())))),
    [([([([()])])])],
    np.array([2, 3, 4]),
    np.array([2, 4, 5]),
    pd.DataFrame({"a": [4, 2]}),
    pd.DataFrame({"a": [4, 3]}),
    (np.array([1, 2, 4]), [pd.DataFrame({"a": [4, 2]})]),
    ForecastingHorizon([1, 2, 3], is_relative=True),
    ForecastingHorizon([1, 2, 3], is_relative=False),
    {"foo": [42], "bar": pd.Series([1, 2])},
    {"bar": [42], "foo": pd.Series([1, 2])},
]


@pytest.mark.parametrize("fixture", EXAMPLES)
def test_deep_equals_positive(fixture):
    """Tests that deep_equals correctly identifies equal objects as equal."""
    x = deepcopy(fixture)
    y = deepcopy(fixture)

    msg = (
        f"deep_copy incorrectly returned False for two identical copies of "
        f"the following object: {x}"
    )
    assert deep_equals(x, y), msg


n = len(EXAMPLES)
DIFFERENT_PAIRS = [
    (EXAMPLES[i], EXAMPLES[j]) for i in range(n) for j in range(n) if i != j
]


@pytest.mark.parametrize("fixture1,fixture2", DIFFERENT_PAIRS)
def test_deep_equals_negative(fixture1, fixture2):
    """Tests that deep_equals correctly identifies unequal objects as unequal."""
    x = deepcopy(fixture1)
    y = deepcopy(fixture2)

    msg = (
        f"deep_copy incorrectly returned True when comparing "
        f"the following, different objects: x={x}, y={y}"
    )
    assert not deep_equals(x, y), msg
