"""Tests for TemporianTransformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.temporian import TemporianTransformer


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_flat_univariate():
    """Tests basic function works on non-indexed univariate time series."""
    X = load_airline()[0:32]

    def function(evset):
        return evset["Number of airline passengers"] + 1

    transformer = TemporianTransformer(function=function)
    X_transformed = transformer.fit_transform(X=X)

    pd.testing.assert_series_equal(X_transformed, X + 1)


# TODO: add more tests (non-exhaustive, off the top of my mind)
# - add support for pd-multiindex and test
# - add support for pd_multiindex_hier and test
# - test dtypes are preserved?
# - test with already-compiled tp function after we auto-compile it in constructor
# - test with more complex Temporian functions
# - error cases:
#   - function doesn't receive single EventSet
#   - function doesn't return single EventSet
#   - function raises
#   - function is not callable
