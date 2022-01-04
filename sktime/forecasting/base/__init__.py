# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements base classes for forecasting in sktime."""

__all__ = ["ForecastingHorizon", "BaseForecaster", "VALID_FORECASTING_HORIZON_TYPES"]

from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._fh import (
    VALID_FORECASTING_HORIZON_TYPES,
    ForecastingHorizon,
)
