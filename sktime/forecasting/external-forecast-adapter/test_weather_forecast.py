"""Unit tests for ExternalForecastAdapter."""

import unittest

import pandas as pd
from external_forecasts import ExternalForecastAdapter

from sktime.forecasting.base import ForecastingHorizon


class TestExternalForecastAdapter(unittest.TestCase):
    """Unit tests for the ExternalForecastAdapter class."""

    def setUp(self):
        """Set up sample data and initialize the forecaster."""
        self.dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        self.y = pd.Series([15, 16, 14, 18, 20, 19, 17, 16, 15, 14], index=self.dates)
        self.forecaster = ExternalForecastAdapter(location="New York", forecast_days=3)

    def test_fit(self):
        """Test the fit method of ExternalForecastAdapter."""
        self.forecaster.fit(self.y)
        self.assertIsNotNone(self.forecaster.y_)

    def test_predict(self):
        """Test the predict method of ExternalForecastAdapter."""
        self.forecaster.fit(self.y)
        fh = ForecastingHorizon([1, 2, 3], is_relative=True)
        forecast = self.forecaster.predict(fh)
        self.assertEqual(len(forecast), 3)  # Ensure it predicts 3 days ahead


if __name__ == "__main__":
    unittest.main()
