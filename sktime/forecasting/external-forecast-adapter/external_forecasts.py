"""Defines the adapter for external forecasting services."""

import pandas as pd

from sktime.forecasting.base import BaseForecaster

"""Module for integrating external forecasts into sktime."""


class ExternalForecasts(BaseForecaster):
    """Forecaster for integrating external APIs (e.g., Weather API)."""

    def __init__(self, provider):
        self.provider = provider
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        self._fh = fh  # Store forecasting horizon
        return self  # No real fitting needed

    def _predict(self, fh=None, X=None):
        """Retrieve external forecasts for the given forecasting horizon."""
        if fh is None:
            fh = self._fh
        if fh is None:
            raise ValueError("Forecasting horizon (fh) must be provided.")

        fh_list = list(fh.to_numpy())  # Convert sktime fh to list
        cutoff = pd.Timestamp.now()  # Dummy cutoff value
        return self.provider.get_forecast(cutoff, fh_list)
