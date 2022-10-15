#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Trend and STL forecaster tests."""

__author__ = ["Markus Löning", "topher-lo"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.trend import PolynomialTrendForecaster, STLForecaster
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils.estimators._forecasters import MockForecaster


@pytest.mark.parametrize(
    "degree,with_intercept",
    [
        (0, True),
        (1, True),
        (1, False),
        (3, True),
        (3, False),
    ],
)
def test_trend(degree, with_intercept):
    """Checks fitted coefficients of the PolynomialTrendForecaster."""
    y = make_forecasting_problem()
    forecaster = PolynomialTrendForecaster(degree=degree, with_intercept=with_intercept)
    forecaster.fit(y)
    # check coefficients
    # intercept is added in reverse order
    result = forecaster.regressor_.steps[-1][1].coef_[::-1]
    poly_matrix = np.vander(np.arange(len(y)), degree + 1)
    if not with_intercept:
        poly_matrix = poly_matrix[:, :-1]
    expected = np.linalg.lstsq(poly_matrix, y.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(result, expected)


def test_constant_trend():
    """Checks PolynomialTrendForecaster predict with constant trend."""
    y = pd.Series(np.arange(30))
    fh = -np.arange(30)  # in-sample fh
    forecaster = PolynomialTrendForecaster(degree=1)
    y_pred = forecaster.fit(y).predict(fh)
    np.testing.assert_array_almost_equal(y, y_pred)


def test_stl_pred_var():
    """Checks the predicted variance of STLForecaster."""
    y = pd.Series(np.arange(30))
    mock_forecaster = MockForecaster(prediction_constant=10)
    forecaster = STLForecaster(
        forecaster_resid=mock_forecaster,
        forecaster_seasonal=mock_forecaster,
        forecaster_trend=mock_forecaster,
    )
    fh = [1, 2, 3, 4]
    y_pred_var = forecaster.fit(y).predict_var(fh=fh)
    expected = mock_forecaster.fit(y).predict_var(fh=fh) * 3
    pd.testing.assert_frame_equal(y_pred_var, expected)
