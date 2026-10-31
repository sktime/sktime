"""
ExternalForecasts integrates external forecast APIs into sktime.

This adapter fetches forecasts from an external provider and uses them
as predicted values in a time-series forecasting workflow.
"""

import pandas as pd
from forecast_provider import ForecastProvider

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA


class ExternalForecasts(BaseForecaster):
    """
    Adapter for integrating external forecast APIs into sktime.

    This forecaster fetches data from an external provider and returns predictions
    based on the external forecast.

    Attributes
    ----------
    api_url : str
        URL of the external forecast API.
    api_key : str
        API key for authentication.
        Handles API requests or external data retrieval.
    """

    _tags = {
        "requires-fh-in-fit": False,
        "non-deterministic": True,
        "capability:pred_int": False,
    }

    def __init__(self, provider: ForecastProvider):
        """Initialize the external forecast adapter."""
        self.provider = provider
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
        self : ExternalForecasts
            Returns the instance itself.
        """
        self.y_ = y
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """
        Predict using external forecast data.

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
        cutoff = self.cutoff
        forecast_values = self.provider.get_forecast(cutoff, fh)
        return forecast_values

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
        self : ExternalForecasts
            Returns the instance itself.
        """
        self.y_ = y
        return self


def example_usage():
    """Demonstrate how to use the ExternalForecasts with sktime."""
    from sktime.forecasting.base import ForecastingHorizon

    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    y = pd.Series([15, 16, 14, 18, 20, 19, 17, 16, 15, 14], index=dates)

    provider = ForecastProvider(
        api_url="https://forecast.example.com/get_forecast", api_key="your_api_key"
    )

    forecaster = ExternalForecasts(provider)

    forecaster.fit(y)
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    forecast = forecaster.predict(fh)

    print("External API forecast:")
    print(forecast)


if __name__ == "__main__":
    example_usage()
