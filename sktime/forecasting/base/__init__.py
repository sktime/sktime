# -*- coding: utf-8 -*-
__all__ = [
    "ForecastingHorizon",
    "BaseForecaster",
    "is_forecaster",
]

from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._base import is_forecaster
