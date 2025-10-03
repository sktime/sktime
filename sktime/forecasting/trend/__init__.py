#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements trend forecasters."""

__all__ = [
    "TrendForecaster",
    "PolynomialTrendForecaster",
    "SplineTrendForecaster",
    "STLForecaster",
    "CurveFitForecaster",
    "ProphetPiecewiseLinearTrendForecaster",
    "SplineTrendForecaster",
]

from sktime.forecasting.trend._curve_fit_forecaster import CurveFitForecaster
from sktime.forecasting.trend._polynomial_trend_forecaster import (
    PolynomialTrendForecaster,
)
from sktime.forecasting.trend._pwl_trend_forecaster import (
    ProphetPiecewiseLinearTrendForecaster,
)
from sktime.forecasting.trend._spline_trend_forecaster import (
    SplineTrendForecaster,
)
from sktime.forecasting.trend._stl_forecaster import STLForecaster
from sktime.forecasting.trend._trend_forecaster import TrendForecaster
