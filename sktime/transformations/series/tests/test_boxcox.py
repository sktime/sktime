# -*- coding: utf-8 -*-
"""Tests for BoxCoxTransformer."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]
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
    "method, sp", [("mle", None), ("pearsonr", None), ("guerrero", 5)]
)
def test_lambda_bounds(bounds, method, sp):
    y = load_airline()
    t = BoxCoxTransformer(bounds=bounds, method=method, sp=sp)
    t.fit(y)
    assert bounds[0] < t.lambda_ < bounds[1]


@pytest.mark.parametrize(
    "bounds, r_lambda",
    [
        ((0, 1), 6.61069613518961e-05),
        ((-1, 0), -0.156975034727656),
        ((-1, 2), -0.156981228426408),
    ],
)
def test_guerrero_against_r_implementation(bounds, r_lambda):
    """Test BoxCoxTransformer against forecast guerrero method.

    Testing lambda values estimated by the R implementation of the Guerrero method
    https://github.com/robjhyndman/forecast/blob/master/R/guerrero.R
    against the guerrero method in BoxCoxTransformer.
    R code to generate the hardcoded value for bounds=(-1, 2) used in the test
    ('Airline.csv' contains the data from 'load_airline()'):
        airline_file <- read.csv(file = 'Airline.csv')[,c('Passengers')]
        airline.ts <- ts(airline_file)
        guerrero(airline.ts, lower=-1, upper=2, nonseasonal.length = 20)
    Output:
        -0.156981228426408
    """
    y = load_airline()
    t = BoxCoxTransformer(bounds=bounds, method="guerrero", sp=20)
    t.fit(y)
    np.testing.assert_almost_equal(t.lambda_, r_lambda, decimal=4)
