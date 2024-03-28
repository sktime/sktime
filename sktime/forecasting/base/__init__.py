# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements base classes for forecasting in sktime."""

__all__ = [
    "ForecastingHorizon",
    "BaseForecaster",
    "GlobalBaseForecaster",
]

from sktime.forecasting.base._base import BaseForecaster, GlobalBaseForecaster
from sktime.forecasting.base._fh import ForecastingHorizon
