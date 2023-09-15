#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements trend forecasters."""

__all__ = [
    "TrendForecaster",
    "PolynomialTrendForecaster",
    "STLForecaster",
]

from forecasting.trend.polynomial_trend_forecaster import PolynomialTrendForecaster
from forecasting.trend.stl_forecaster import STLForecaster
from forecasting.trend.trend_forecaster import TrendForecaster
