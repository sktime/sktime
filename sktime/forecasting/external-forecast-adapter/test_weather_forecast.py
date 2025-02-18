"""Unit tests for the weather forecast provider integration."""

import pandas as pd
from external_forecasts import ExternalForecasts
from weather_forecast_provider import WeatherForecastProvider

from sktime.forecasting.base._fh import ForecastingHorizon

# Create provider and forecaster
provider = WeatherForecastProvider()
weather_forecaster = ExternalForecasts(provider)

# Define forecasting horizon (e.g., next 3 days)
fh = ForecastingHorizon([1, 2, 3], is_relative=True)

# Fit the forecaster (dummy fit)
weather_forecaster.fit(y=pd.Series([20, 21, 22]), fh=fh)

# Predict
forecast = weather_forecaster.predict(fh)
print(forecast)
