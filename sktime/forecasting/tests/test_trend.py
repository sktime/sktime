#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pytest
from sktime.datasets import load_airline
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformers.single_series.detrend import Detrender


def compute_expected_coefs(y, degree, with_intercept=True):
    """Helper function to compute expected coefficients from polynomial
    regression"""
    poly_matrix = np.vander(y.index.values, degree + 1)
    if not with_intercept:
        poly_matrix = poly_matrix[:, :-1]
    return np.linalg.lstsq(poly_matrix, y.values, rcond=None)[0]


def compute_expected_detrend(y, degree, with_intercept):
    a, b = compute_expected_coefs(y, degree, with_intercept)
    yt = np.zeros(len(y))
    for i in range(len(y)):
        yt[i] = y[i] - ((a * i) + b)
    return yt


def check_trend(degree, with_intercept):
    """Helper function to check trend"""
    y = load_airline()
    f = PolynomialTrendForecaster(degree=degree, with_intercept=with_intercept)
    f.fit(y)
    a = f.regressor_.steps[-1][1].coef_[
        ::-1]  # intercept is added in reverse order

    b = compute_expected_coefs(y, degree, with_intercept)
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("with_intercept", [True, False])
def test_trend(degree, with_intercept):
    check_trend(degree, with_intercept)


@pytest.mark.parametrize("degree", [0])
def test_zero_trend(degree, with_intercept=True):
    # zero trend does not work without intercept, hence separate test
    check_trend(degree, with_intercept)


def test_linear_detrending():
    y = load_airline()

    f = PolynomialTrendForecaster(degree=1, with_intercept=True)
    t = Detrender(f)
    a = t.fit_transform(y)

    b = compute_expected_detrend(y, 1, with_intercept=True)

    np.testing.assert_allclose(a, b)
