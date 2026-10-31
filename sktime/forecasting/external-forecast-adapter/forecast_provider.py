"""A common interface for all external forecast providers."""

from abc import ABC, abstractmethod

import pandas as pd


class ForecastProvider(ABC):
    """Abstract class for external forecast providers."""

    @abstractmethod
    def get_forecast(self, cutoff, fh) -> pd.Series:
        """
        Fetch forecasts from an external source.

        Parameters
        ----------
        - cutoff (pd.Timestamp): Last known data point.
        - fh (ForecastingHorizon): Forecasting horizon.

        Returns
        -------
        - pd.Series: Forecasted values indexed by forecast horizon.
        """
        pass
