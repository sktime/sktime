"""Unit tests for ExternalForecastAdapter."""

import pandas as pd

from sktime.forecasting.model_selection import temporal_train_test_split

from .api_forecast_provider import APIForecastProvider
from .external_forecasts import ExternalForecasts

# Example data
y = pd.Series(
    [10, 20, 30, 40, 50], index=pd.date_range("2023-01-01", periods=5, freq="D")
)
y_train, y_test = temporal_train_test_split(y, test_size=2)

# Example API URL (replace with actual API)
api_url = "https://forecast.example.com/get_forecast"

# Create API provider
provider = APIForecastProvider(api_url=api_url, params={"location": "New York"})

# Create the ExternalForecasts model
forecaster = ExternalForecasts(provider)

# No need to fit
forecaster.fit(y_train)

# Predict using external forecasts
fh = [1, 2]  # Forecasting horizon
y_pred = forecaster.predict(fh=fh)

print("Predicted values:", y_pred)
