"""Tests for TabularToSeriesAdaptor."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]

import numpy as np
import pytest
from sklearn.preprocessing import PowerTransformer

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.adapt import TabularToSeriesAdaptor


@pytest.mark.skipif(
    not run_test_for_class([TabularToSeriesAdaptor, PowerTransformer]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_boxcox_transform():
    """Test whether adaptor based transformer behaves like the raw wrapped method."""
    from scipy.stats import boxcox

    y = load_airline()
    t = TabularToSeriesAdaptor(PowerTransformer(method="box-cox", standardize=False))
    actual = t.fit_transform(y)

    expected, _ = boxcox(np.asarray(y))  # returns fitted lambda as second output
    np.testing.assert_array_equal(actual, expected)
