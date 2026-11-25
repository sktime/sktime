"""
Module for integrating external weather forecast APIs into sktime.

This module provides an adapter that allows using external weather data
as a forecaster for time-series analysis.
"""

from .external_forecasts import ExternalForecastAdapter
from .weather_forecast_provider import WeatherForecastProvider

__all__ = ["ExternalForecastAdapter", "WeatherForecastProvider"]
