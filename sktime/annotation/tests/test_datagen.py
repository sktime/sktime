# -*- coding: utf-8 -*-
"""Tests for datagen functions."""
__author__ = ["klam-data", "mgorlin", "pyyim"]

import pytest

from sktime.annotation.datagen import piecewise_poisson


@pytest.mark.parametrize(
    "lambdas, lengths, random_state, output",
    [
        ([1, 2, 3], [2, 4, 8], 42, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        ([1, 3, 6], [2, 4, 8], 42, [2, 2, 2, 2, 2, 2, 1, 0, 1, 0, 1, 0, 1, 0]),
    ],
)
def test_piecewise_poisson(lambdas, lengths, random_state, output):
    """Test piecewise_poisson fuction returns the expected Poisson distributed array."""
    assert piecewise_poisson(lambdas, lengths, random_state) == output
