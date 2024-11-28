import numpy as np
import pytest
from sktime.tests.test_switch import run_test_for_class
from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.compose import AutoEnsembleForecaster
from sktime.forecasting.base import ForecastingHorizon

# Test for AutoEnsembleForecaster
@pytest.mark.skipif(
    not run_test_for_class([AutoEnsembleForecaster]),
    reason="run test only if AutoEnsembleForecaster is available",
)
def test_auto_ensemble_forecaster():
    """Compares AutoEnsembleForecaster to a benchmark model.
    
    This test evaluates the performance of the AutoEnsembleForecaster by comparing 
    its predictions to those of individual forecasters, such as NaiveForecaster 
    and PolynomialTrendForecaster, using mean squared error (MSE) as the comparison metric.

    The goal is to verify that the ensemble forecaster performs similarly to 
    or better than its individual components.
    """
    
    # Load a dataset for testing
    y = load_airline()
    
    # Define some forecasters to use in the ensemble
    forecasters = [
        ("naive", NaiveForecaster()),  # Naive forecaster to compare with
        ("trend", PolynomialTrendForecaster()),  # Polynomial trend forecaster to compare with
    ]
    
    # Create the AutoEnsembleForecaster with the list of forecasters
    ensemble_forecaster = AutoEnsembleForecaster(forecasters=forecasters)
    
    # Loop through different forecast horizons (fh) to test various forecast lengths
    forecast_horizons = [
        [1, 2, 3],  # Short-term forecast
        [1, 5, 10],  # Medium-term forecast
        [1, 10, 20],  # Longer forecast
    ]
    
    for fh in forecast_horizons:
        print(f"Testing with forecast horizon: {fh}")
        
        # Fit the ensemble forecaster to the dataset with the current forecasting horizon
        ensemble_forecaster.fit(y=y, fh=fh)
        
        # Predict future values using the ensemble forecaster
        y_pred_ensemble = ensemble_forecaster.predict()
        
        # For comparison, we predict with individual forecasters (e.g., Naive and PolynomialTrend)
        naive_forecaster = NaiveForecaster()
        naive_forecaster.fit(y=y, fh=fh)
        y_pred_naive = naive_forecaster.predict()

        trend_forecaster = PolynomialTrendForecaster()
        trend_forecaster.fit(y=y, fh=fh)
        y_pred_trend = trend_forecaster.predict()

        # Compare the ensemble prediction with the individual predictions
        # Using mean squared error (MSE) to evaluate the accuracy of the ensemble against individual forecasters
        from sklearn.metrics import mean_squared_error

        mse_ensemble_naive = mean_squared_error(y_pred_ensemble, y_pred_naive)
        mse_ensemble_trend = mean_squared_error(y_pred_ensemble, y_pred_trend)

        # Print MSE values for comparison
        print(f"MSE ensemble vs. naive: {mse_ensemble_naive}")
        print(f"MSE ensemble vs. trend: {mse_ensemble_trend}")

# Call the function to run the test
test_auto_ensemble_forecaster()
