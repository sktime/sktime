"""WeatherForecastProvider fetches forecast data from an external weather API."""

import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class WeatherForecastProvider:
    """
    Fetch weather forecast data from an external API.

    Attributes
    ----------
    BASE_URL : str
        API endpoint for fetching weather data.
    api_key : str
        API key loaded from environment variables.
    """

    BASE_URL = "https://api.weatherapi.com/v1/forecast.json"  # Example API endpoint

    def __init__(self):
        """Initialize the provider and load the API key."""
        self.api_key = os.getenv("WEATHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing API Key! Set WEATHER_API_KEY in environment variables."
            )

    def get_forecast(self, location: str, days: int = 1):
        """
        Fetch weather forecast for a given location.

        Parameters
        ----------
        location : str
            City name or coordinates.
        days : int, optional
            Number of days to forecast (default: 1).

        Returns
        -------
        dict
            Weather data in JSON format.

        Raises
        ------
        Exception
            If the API request fails.
        """
        params = {
            "key": self.api_key,
            "q": location,
            "days": days,
        }

        response = requests.get(self.BASE_URL, params=params)
        if response.status_code != 200:
            error_msg = (
                f"API request failed! Status: {response.status_code}, "
                f"Response: {response.text[:75]}..."
            )
            raise Exception(error_msg)

        return response.json()
