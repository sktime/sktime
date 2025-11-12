"""Example usage of ETSForecaster class.
"""

from sktime.datasets import load_airline
from sktime.forecasting.ets import ETSForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.plotting import plot_series

# Load and split data
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=12)

# Fit and predict
forecaster = ETSForecaster(m=12, model="ZZZ")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh=list(range(1, 13)))
pred_int = forecaster.predict_interval(fh=list(range(1, 13)))

y_pred

# Check the selected model forecast title
selected_model = f"{forecaster.model_.config.error}{forecaster.model_.config.trend}"
if forecaster.model_.config.damped:
    selected_model += "d"

selected_model += forecaster.model_.config.season

ptitle = f"Forecast from ETS({selected_model})"

# Plot results
plot_series(y_train, y_test, y_pred, labels=["Train", "Test", "Forecast"], title=ptitle)


# Check the selected model
selected_model = f"{forecaster.model_.config.error}{forecaster.model_.config.trend}"
if forecaster.model_.config.damped:
    selected_model += "d"

selected_model += forecaster.model_.config.season

print(f"Selected model: ETS({selected_model})")

# More detailed information
print(f"Error type: {forecaster.model_.config.error}")
print(f"Trend type: {forecaster.model_.config.trend}")
print(f"Season type: {forecaster.model_.config.season}")
print(f"Damped: {forecaster.model_.config.damped}")
print(f"Seasonal period (m): {forecaster.model_.config.m}")

# Model parameters
print(f"\nEstimated parameters:")
print(f"  alpha (level): {forecaster.model_.params.alpha:.4f}")
if forecaster.model_.config.trend != "N":
    print(f"  beta (trend): {forecaster.model_.params.beta:.4f}")

if forecaster.model_.config.season != "N":
    print(f"  gamma (seasonal): {forecaster.model_.params.gamma:.4f}")
if forecaster.model_.config.damped:
    print(f"  phi (damping): {forecaster.model_.params.phi:.4f}")

# Model fit statistics
print(f"\nModel fit:")
print(f"  AIC: {forecaster.model_.aic:.2f}")
print(f"  BIC: {forecaster.model_.bic:.2f}")
print(f"  Log-likelihood: {forecaster.model_.loglik:.2f}")
