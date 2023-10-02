"""Tests for TabularToSeriesAdaptor."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]

import numpy as np
import pytest
from sklearn.preprocessing import PowerTransformer

from sktime.datasets import load_airline
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("scipy", severity="none"),
    reason="skip test if required soft dependencies not available",
)
def test_boxcox_transform():
    """Test whether adaptor based transformer behaves like the raw wrapped method."""
    from scipy.stats import boxcox

    y = load_airline()
    t = TabularToSeriesAdaptor(PowerTransformer(method="box-cox", standardize=False))
    actual = t.fit_transform(y)

    expected, _ = boxcox(np.asarray(y))  # returns fitted lambda as second output
    np.testing.assert_array_equal(actual, expected)
