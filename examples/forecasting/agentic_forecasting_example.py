# Author: Mishti Agarwal
# This example demonstrates agentic-style forecasting using sktime
"""
Agentic forecasting example using sktime.

This example demonstrates a simple workflow:
1. Train a forecasting model
2. Generate predictions
3. Provide a simple explanation (agentic-style output)
"""

from sktime.datasets import load_airline
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.base import ForecastingHorizon


def explain_forecast(pred):
    """Generate a simple explanation for forecast output."""
    return f"The forecast indicates future values: {list(pred)}"


# Load dataset
y = load_airline()

# Define forecasting horizon
fh = ForecastingHorizon([1, 2, 3], is_relative=True)

# Train model
forecaster = ThetaForecaster(sp=12)
forecaster.fit(y)

# Predict
y_pred = forecaster.predict(fh)

# Output results
print("Predictions:", y_pred)
print("Explanation:", explain_forecast(y_pred))