#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements trend forecasters."""

__all__ = [
    "TrendForecaster",
    "PolynomialTrendForecaster",
    "STLForecaster",
    "CurveFitForecaster",
]

from sktime.forecasting.trend._curve_fit_forecaster import CurveFitForecaster
from sktime.forecasting.trend._polynomial_trend_forecaster import (
    PolynomialTrendForecaster,
)
from sktime.forecasting.trend._stl_forecaster import STLForecaster
from sktime.forecasting.trend._trend_forecaster import TrendForecaster
