"""Example usage of ETSForecaster class.
"""

import numpy as np
import pandas as pd


"""Simple minimal example of ARAR forecaster usage."""

from sktime.datasets import load_airline
from sktime.forecasting.ets import ETSForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.plotting import plot_series

# Load and split data
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=12)

# Fit and predict
forecaster = ETSForecaster(m=12, model="MAM", damped=True)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh=list(range(1, 13)))
pred_int = forecaster.predict_interval(fh=list(range(1, 13)))

# Plot results
plot_series(y_train, y_test, y_pred, labels=["Train", "Test", "Forecast"])

