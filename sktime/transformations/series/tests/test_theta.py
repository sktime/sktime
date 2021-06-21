#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Guzal Bulatova"]
__all__ = []

import numpy as np
from scipy.stats import linregress

from sktime.datasets import load_airline
from sktime.transformations.series.theta import ThetaLinesTransformer


def test_theta_0():
    y = load_airline()
    t = ThetaLinesTransformer(0)
    t.fit(y)
    actual = t.transform(y)
    x = np.arange(y.size) + 1
    lin_regress = linregress(x, y)
    expected_0 = lin_regress.intercept + lin_regress.slope * x

    np.testing.assert_array_equal(actual, expected_0)


def test_theta_1():
    y = load_airline()
    t = ThetaLinesTransformer(1)
    t.fit(y)
    actual = t.transform(y)

    np.testing.assert_array_equal(actual, y)
