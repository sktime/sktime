#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests of ThetaLinesTransformer functionality."""

__author__ = ["Guzal Bulatova"]
__all__ = []

import numpy as np
import pytest
from scipy.stats import linregress

from sktime.datasets import load_airline
from sktime.transformations.series.theta import ThetaLinesTransformer


def test_theta_0():
    # with theta = 0
    y = load_airline()
    t = ThetaLinesTransformer(0)
    t.fit(y)
    actual = t.transform(y)
    x = np.arange(y.size) + 1
    lin_regress = linregress(x, y)
    expected = lin_regress.intercept + lin_regress.slope * x

    np.testing.assert_almost_equal(actual, expected, decimal=8)


def test_theta_1():
    # with theta = 1 Theta-line is equal to the original time-series
    y = load_airline()
    t = ThetaLinesTransformer(1)
    t.fit(y)
    actual = t.transform(y)
    np.testing.assert_array_equal(actual, y)


@pytest.mark.parametrize("theta", [(1, 1.5), (0, 1, 2), (0.25, 0.5, 0.75, 1, 2)])
def test_thetalines_shape(theta):
    y = load_airline()
    t = ThetaLinesTransformer(theta)
    t.fit(y)
    actual = t.transform(y)
    assert actual.shape == (y.shape[0], len(theta))
