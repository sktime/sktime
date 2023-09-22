# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements base classes for forecasting in sktime."""

__all__ = [
    "ForecastingHorizon",
    "BaseForecaster",
]

from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._fh import ForecastingHorizon
