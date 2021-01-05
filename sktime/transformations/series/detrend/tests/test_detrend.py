#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = []


def test_polynomial_detrending():
    import numpy as np
    import pandas as pd

    from sktime.forecasting.tests.test_trend import get_expected_polynomial_coefs
    from sktime.forecasting.trend import PolynomialTrendForecaster
    from sktime.transformations.series.detrend import Detrender

    y = pd.Series(np.arange(20) * 0.5) + np.random.normal(0, 1, size=20)
    forecaster = PolynomialTrendForecaster(degree=1, with_intercept=True)
    transformer = Detrender(forecaster)
    transformer.fit(y)

    # check coefficients
    actual_coefs = transformer.forecaster_.regressor_.steps[-1][-1].coef_
    expected_coefs = get_expected_polynomial_coefs(y, degree=1, with_intercept=True)[
        ::-1
    ]
    np.testing.assert_array_almost_equal(actual_coefs, expected_coefs)

    # check trend
    expected_trend = expected_coefs[0] + np.arange(len(y)) * expected_coefs[1]
    actual_trend = transformer.forecaster_.predict(-np.arange(len(y)))
    np.testing.assert_array_almost_equal(actual_trend, expected_trend)

    # check residuals
    actual = transformer.transform(y)
    expected = y - expected_trend
    np.testing.assert_array_almost_equal(actual, expected)
