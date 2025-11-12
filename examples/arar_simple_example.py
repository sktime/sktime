"""Simple minimal example of ARAR forecaster usage."""

from sktime.datasets import load_airline
from sktime.forecasting.arar import ARARForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.plotting import plot_series

# Load and split data
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=12)

# Fit and predict
forecaster = ARARForecaster()
forecaster.fit(y_train)
y_pred = forecaster.predict(fh=list(range(1, 13)))
pred_int = forecaster.predict_interval(fh=list(range(1, 13)))

# Plot results
plot_series(y_train, y_test, y_pred, labels=["Train", "Test", "Forecast"])

# Print model information
print(f"Selected AR lags: {forecaster.model_[2]}")
print(f"AR coefficients: {forecaster.model_[1]}")
print(f"Innovation variance: {forecaster.model_[3]:.4f}")
