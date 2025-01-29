import pandas as pd
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import RecursiveReductionForecaster

# Define a sample time series
y = pd.Series([1, 2, 3, 4, 5, 6, 7])

# Use a compatible sklearn regressor
base_regressor = LinearRegression()
forecaster = RecursiveReductionForecaster(base_regressor)

# Fit the forecaster
forecaster.fit(y)

# Make predictions
fh = [1, 2, 3]  # Forecast horizon
predictions = forecaster.predict(fh)
print(predictions)
