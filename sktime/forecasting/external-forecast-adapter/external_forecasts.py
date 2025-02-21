"""
ExternalForecastAdapter integrates external weather forecast APIs into sktime.

This adapter fetches weather forecasts from a weather API and uses them
as predicted values in a time-series forecasting workflow.
"""

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA

from .weather_forecast_provider import WeatherForecastProvider


class ExternalForecastAdapter(BaseForecaster):
    """
    Adapter for integrating external weather forecast APIs into sktime.

    This forecaster fetches weather data from an external API and returns predictions
    based on the weather forecast.

    Attributes
    ----------
    location : str
        City or coordinates for fetching weather data.
    forecast_days : int
        Number of days to predict.
    provider : WeatherForecastProvider
        Handles API requests.
    """

    def __init__(self, location: str, forecast_days: int = 1):
        """Initialize the external forecast adapter."""
        self.location = location
        self.forecast_days = forecast_days
        self.provider = WeatherForecastProvider()
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """
        Fit method (required by sktime forecasters).

        This method doesn't train a model but prepares data.

        Parameters
        ----------
        y : pd.Series
            Target time series.
        X : pd.DataFrame, optional
            Exogenous data.
        fh : ForecastingHorizon, optional
            Forecasting horizon.

        Returns
        -------
        self : ExternalForecastAdapter
            Returns the instance itself.
        """
        self.y_ = y
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """
        Predict using external weather data.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame, optional
            Exogenous data.
        return_pred_int : bool, default=False
            Return prediction intervals.
        alpha : float, default=DEFAULT_ALPHA
            Significance level for prediction intervals.

        Returns
        -------
        pd.Series
            Forecasted values.
        """
        forecast_data = self.provider.get_forecast(
            self.location, days=self.forecast_days
        )

        # Extract relevant weather forecast (e.g., temperature)
        forecast_values = [
            forecast_data["forecast"]["forecastday"][i]["day"]["avgtemp_c"]
            for i in range(self.forecast_days)
        ]

        index = pd.date_range(
            start=self.y_.index[-1] + pd.Timedelta(days=1),
            periods=self.forecast_days,
            freq="D",
        )

        return pd.Series(forecast_values, index=index)

    def _update(self, y, X=None, update_params=True):
        """
        Update method (if needed for rolling forecasts).

        Parameters
        ----------
        y : pd.Series
            Updated target time series.
        X : pd.DataFrame, optional
            Updated exogenous data.
        update_params : bool, default=True
            Whether to update parameters.

        Returns
        -------
        self : ExternalForecastAdapter
            Returns the instance itself.
        """
        self.y_ = y
        return self


def example_usage():
    """Demonstrate how to use the ExternalForecastAdapter with sktime."""
    from sktime.forecasting.base import ForecastingHorizon

    # Sample historical temperature data
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    y = pd.Series([15, 16, 14, 18, 20, 19, 17, 16, 15, 14], index=dates)

    # Initialize forecaster
    forecaster = ExternalForecastAdapter(location="New York", forecast_days=3)

    # Fit and predict
    forecaster.fit(y)
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    forecast = forecaster.predict(fh)

    print("Weather-based forecast:")
    print(forecast)


if __name__ == "__main__":
    example_usage()
