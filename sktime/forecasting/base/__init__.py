# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements base classes for forecasting in sktime."""

__all__ = [
    "ForecastingHorizon",
    "BaseForecaster",
    "_BaseGlobalForecaster",
    "BaseProbaForecaster",
]

from sktime.forecasting.base._base import BaseForecaster, _BaseGlobalForecaster
from sktime.forecasting.base._base_proba import BaseProbaForecaster
from sktime.forecasting.base._fh import ForecastingHorizon
