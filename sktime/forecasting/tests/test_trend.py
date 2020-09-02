#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["get_expected_polynomial_coefs"]

import numpy as np
import pytest

from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.forecasting.trend import PolynomialTrendForecaster


def get_expected_polynomial_coefs(y, degree, with_intercept=True):
    """Helper function to compute expected coefficients from polynomial
    regression"""
    poly_matrix = np.vander(np.arange(len(y)), degree + 1)
    if not with_intercept:
        poly_matrix = poly_matrix[:, :-1]
    return np.linalg.lstsq(poly_matrix, y.to_numpy(), rcond=None)[0]


def _test_trend(degree, with_intercept):
    """Helper function to check trend"""
    y = make_forecasting_problem()
    forecaster = PolynomialTrendForecaster(
        degree=degree, with_intercept=with_intercept)
    forecaster.fit(y)

    # check coefficients
    # intercept is added in reverse order
    actual = forecaster.regressor_.steps[-1][1].coef_[::-1]
    expected = get_expected_polynomial_coefs(y, degree, with_intercept)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("with_intercept", [True, False])
def test_trend(degree, with_intercept):
    _test_trend(degree, with_intercept)


# zero trend does not work without intercept
def test_zero_trend():
    _test_trend(degree=0, with_intercept=True)
