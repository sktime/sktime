"""ApiForecastProvider fetches forecast data from an external API."""

import pandas as pd
import requests

from .forecast_provider import ForecastProvider


class APIForecastProvider(ForecastProvider):
    """Fetch forecasts from an external API."""

    def __init__(self, api_url, params=None, headers=None):
        """
        Initialize the API forecast provider.

        Parameters
        ----------
        - api_url (str): API endpoint for fetching forecasts.
        - params (dict, optional): Additional query parameters.
        - headers (dict, optional): HTTP headers (e.g., for authentication).
        """
        self.api_url = api_url
        self.params = params or {}
        self.headers = headers or {}

    def get_forecast(self, cutoff, fh):
        """Fetch forecast data from the API."""
        request_params = {**self.params, "cutoff": str(cutoff), "fh": list(fh)}
        response = requests.get(
            self.api_url, params=request_params, headers=self.headers
        )
        response.raise_for_status()  # Raise error for bad responses
        data = response.json()

        # Ensure forecast values are in a pandas Series
        return pd.Series(data["forecasts"], index=fh.to_absolute(cutoff))
