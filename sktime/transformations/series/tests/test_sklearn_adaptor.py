# -*- coding: utf-8 -*-
"""Tests for TabularToSeriesAdaptor."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]

import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

from sktime.datasets import load_airline
from sktime.transformations.series.adapt import TabularToSeriesAdaptor


def test_boxcox_transform():
    """Test whether adaptor based transformer behaves like the raw wrapped method."""
    y = load_airline()
    t = TabularToSeriesAdaptor(PowerTransformer(method="box-cox", standardize=False))
    actual = t.fit_transform(y)

    expected, _ = boxcox(np.asarray(y))  # returns fitted lambda as second output
    np.testing.assert_array_equal(actual, expected)
