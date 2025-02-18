"""Implements the weather forecast provider interface."""

import pandas as pd
import requests


class WeatherForecastProvider:
    """Provides weather forecast data from an external API."""

    def get_forecast(self, cutoff: pd.Timestamp, fh):
        """Fetch weather forecasts from an external API."""
        api_key = "632a98356425485c8b8182343251802"  # Replace with your API key
        location = "New York"
        url = f"https://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days=3"

        response = requests.get(url)
        data = response.json()

        # Ensure API response contains expected data
        if "forecast" not in data:
            raise ValueError("API response does not contain 'forecast' key.")

        fh_list = list(fh)  # Convert forecasting horizon to a list

        # Extract forecast values
        forecast_values = {
            h: data["forecast"]["forecastday"][0]["day"]["avgtemp_c"] for h in fh_list
        }

        return pd.Series(forecast_values, index=fh_list)
