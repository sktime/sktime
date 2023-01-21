# -*- coding: utf-8 -*-
"""Test trend forecasters."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["mloning"]
__all__ = ["get_expected_polynomial_coefs"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.trend import PolynomialTrendForecaster, TrendForecaster
from sktime.utils._testing.forecasting import make_forecasting_problem


def get_expected_polynomial_coefs(y, degree, with_intercept=True):
    """Compute expected coefficients from polynomial regression."""
    poly_matrix = np.vander(np.arange(len(y)), degree + 1)
    if not with_intercept:
        poly_matrix = poly_matrix[:, :-1]
    return np.linalg.lstsq(poly_matrix, y.to_numpy(), rcond=None)[0]


def _test_trend(degree, with_intercept):
    """Check trend, helper function."""
    y = make_forecasting_problem()
    forecaster = PolynomialTrendForecaster(degree=degree, with_intercept=with_intercept)
    forecaster.fit(y)

    # check coefficients
    # intercept is added in reverse order
    actual = forecaster.regressor_.steps[-1][1].coef_[::-1]
    expected = get_expected_polynomial_coefs(y, degree, with_intercept)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("with_intercept", [True, False])
def test_trend(degree, with_intercept):
    """Test PolynomialTrendForecaster coefficients."""
    _test_trend(degree, with_intercept)


# zero trend does not work without intercept
def test_zero_trend():
    """Test PolynomialTrendForecaster with degree zero."""
    _test_trend(degree=0, with_intercept=True)


def test_constant_trend():
    """Test expected output from constant trend."""
    y = pd.Series(np.arange(30))
    fh = -np.arange(30)  # in-sample fh

    forecaster = PolynomialTrendForecaster(degree=1)
    y_pred = forecaster.fit(y).predict(fh)

    np.testing.assert_array_almost_equal(y, y_pred)


def test_trendforecaster_with_datetimeindex():
    """Test PolyonmialTrendForecaster with DatetimeIndex, see #4131."""
    df = load_airline()
    df.index = df.index.to_timestamp()

    f = PolynomialTrendForecaster()
    f.fit(df)

    f = TrendForecaster()
    f.fit(df)
