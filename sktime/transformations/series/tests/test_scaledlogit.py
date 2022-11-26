# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""ScaledLogit transform unit tests."""

__author__ = ["ltsaprounis"]

from warnings import warn

import numpy as np
import pytest

from sktime.datasets import load_airline
from sktime.transformations.series.scaledlogit import ScaledLogitTransformer

TEST_SERIES = np.array([30, 40, 60])


@pytest.mark.parametrize(
    "lower, upper, output",
    [
        (10, 70, np.log((TEST_SERIES - 10) / (70 - TEST_SERIES))),
        (None, 70, -np.log(70 - TEST_SERIES)),
        (10, None, np.log(TEST_SERIES - 10)),
        (None, None, TEST_SERIES),
    ],
)
def test_scaledlogit_transform(lower, upper, output):
    """Test that we get the right output."""
    transformer = ScaledLogitTransformer(lower, upper)
    y_transformed = transformer.fit_transform(TEST_SERIES)
    assert np.all(output == y_transformed)


@pytest.mark.parametrize(
    "lower, upper, message",
    [
        (
            0,
            300,
            (
                "X in ScaledLogitTransformer should not have values greater"
                "than upper_bound"
            ),
        ),
        (
            300,
            700,
            "X in ScaledLogitTransformer should not have values lower than lower_bound",
        ),
    ],
)
def test_scaledlogit_bound_errors(lower, upper, message):
    """Tests all exceptions."""
    y = load_airline()
    with pytest.warns(RuntimeWarning):
        ScaledLogitTransformer(lower, upper).fit_transform(y)
        warn(message, RuntimeWarning)
