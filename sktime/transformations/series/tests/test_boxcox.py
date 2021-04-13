#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pytest
from scipy.stats import boxcox
from sktime.datasets import load_airline
from sktime.transformations.series.boxcox import BoxCoxTransformer


def test_boxcox_against_scipy():
    y = load_airline()

    t = BoxCoxTransformer()
    actual = t.fit_transform(y)

    excepted, expected_lambda = boxcox(y.values)

    np.testing.assert_array_equal(actual, excepted)
    assert t.lambda_ == expected_lambda


@pytest.mark.parametrize("bounds", [(0, 1), (-1, 0), (-1, 2)])
@pytest.mark.parametrize(
    "method_sp", [("mle", None), ("pearsonr", None), ("guerrero", 5)]
)
def test_lambda_bounds(bounds, method_sp):
    y = load_airline()
    t = BoxCoxTransformer(bounds=bounds, method=method_sp[0], sp=method_sp[1])
    t.fit(y)
    assert bounds[0] < t.lambda_ < bounds[1]
