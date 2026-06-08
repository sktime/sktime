"""Tests for datagen functions."""

__author__ = ["klam-data", "mgorlin", "pyyim"]

import pytest
from numpy import array_equal

from sktime.detection.datagen import piecewise_poisson
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(piecewise_poisson),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "lambdas, lengths, random_state, output",
    [
        ([1, 2, 3], [2, 4, 8], 42, [1, 2, 1, 3, 3, 1, 3, 1, 3, 2, 2, 4, 2, 1]),
        ([1, 3, 6], [2, 4, 8], 42, [1, 2, 1, 3, 3, 2, 5, 5, 6, 4, 4, 9, 3, 5]),
    ],
)
def test_piecewise_poisson(lambdas, lengths, random_state, output):
    """Test piecewise_poisson function returns expected Poisson distributed array."""
    assert array_equal(piecewise_poisson(lambdas, lengths, random_state), output)
