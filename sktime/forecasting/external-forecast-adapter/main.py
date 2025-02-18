"""Main entry point for external forecast integration."""

# main.py
from external_forecasts import ExternalForecasts
from weather_forecast_provider import WeatherForecastProvider

# Instantiate with a weather API provider
weather_provider = WeatherForecastProvider()
weather_forecaster = ExternalForecasts(provider=weather_provider)

# Define forecasting horizon
fh = [1, 2, 3]  # Next 3 time points

# Predict using external data
forecast = weather_forecaster.predict(fh)
print("Weather Forecast:", forecast)
