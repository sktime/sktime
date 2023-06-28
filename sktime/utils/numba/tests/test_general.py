"""Tests for numba functions."""

__author__ = ["TonyBagnall"]

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sktime.utils.validation._dependencies import _check_soft_dependencies

DATATYPES = ["int32", "int64", "float32", "float64"]


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("type", DATATYPES)
def test_z_normalise_series(type):
    """Test the function z_normalise_series."""
    from sktime.utils.numba.general import z_normalise_series

    a = np.array([2, 2, 2], dtype=type)
    a_expected = np.array([0, 0, 0], dtype=type)
    a_result = z_normalise_series(a)
    assert_array_equal(a_result, a_expected)
