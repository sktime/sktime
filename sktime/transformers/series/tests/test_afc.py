#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-


import numpy as np
from statsmodels.tsa.stattools import acf
from sktime.datasets import load_airline
from sktime.transformers.series.acf import AutoCorrelationFunctionTransformer


def test_acf():
    y = load_airline()

    t = AutoCorrelationFunctionTransformer()
    actual = t.fit_transform(y)

    excepted, _ = acf(y.values)

    np.testing.assert_array_equal(actual, excepted)
