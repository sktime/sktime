# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Non-suite tests for probability distribution objects."""

__author__ = ["fkiraly"]

import pytest

from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tensorflow_probability", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_proba_example():
    """Test one subsetting case for BaseDistribution."""
    from sktime.proba.tfp import TFNormal

    n = TFNormal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)

    assert n.shape == (3, 2)

    one_row = n.loc[[1]]
    assert isinstance(one_row, TFNormal)
    assert one_row.shape == (1, 2)
