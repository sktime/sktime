import pytest
from sktime.datasets import load_airline
from sktime.forecasting.ets import AutoETS
import pandas as pd
import numpy as np

@pytest.fixture
def airline_data():
    """Fixture to load airline data once for all tests"""
    return load_airline()

@pytest.fixture
def fitted_forecaster(airline_data):
    """Fixture to create and fit an AutoETS forecaster"""
    forecaster = AutoETS(auto=True, n_jobs=-1)
    forecaster.fit(airline_data)
    return forecaster

def test_predict_quantiles_short_horizon(fitted_forecaster):
    """Test predict_quantiles with short forecast horizon"""
    fh = [1, 2, 3]
    quantiles_result = fitted_forecaster.predict_quantiles(fh=fh)
    
    # Check if result is a DataFrame
    assert isinstance(quantiles_result, pd.DataFrame)
    # Check if number of rows matches forecast horizon length
    assert len(quantiles_result) == len(fh)
    # Check if quantiles columns exist (typically 0.025 and 0.975)
    assert len(quantiles_result.columns) >= 2

def test_predict_quantiles_single_point(fitted_forecaster):
    """Test predict_quantiles with single point horizon"""
    fh = [3]
    quantiles_result = fitted_forecaster.predict_quantiles(fh=fh)
    
    assert isinstance(quantiles_result, pd.DataFrame)
    assert len(quantiles_result) == 1
    assert len(quantiles_result.columns) >= 2

def test_predict_quantiles_long_horizon(fitted_forecaster):
    """Test predict_quantiles with longer horizons that might fail"""
    fh = [10, 100, 1000]
    try:
        quantiles_result = fitted_forecaster.predict_quantiles(fh=fh)
        assert isinstance(quantiles_result, pd.DataFrame)
        assert len(quantiles_result) == len(fh)
    except ValueError as e:
        # Check if we get the expected ValueError for incompatible shapes
        assert "Shape of passed values" in str(e)

def test_predict_quantiles_mixed_horizon(fitted_forecaster):
    """Test predict_quantiles with mixed positive horizons"""
    fh = [1, 5, 10]
    quantiles_result = fitted_forecaster.predict_quantiles(fh=fh)
    
    assert isinstance(quantiles_result, pd.DataFrame)
    assert len(quantiles_result) == len(fh)

def test_predict_quantiles_one_int_horizon(fitted_forecaster):
    """Test predict_quantiles with mixed positive horizons"""
    fh=1
    quantiles_result = fitted_forecaster.predict_quantiles(fh=fh)
    
    assert isinstance(quantiles_result, pd.DataFrame)

def test_predict_quantiles_int_horizon(fitted_forecaster):
    """Test predict_quantiles with mixed positive horizons"""
    fh=5
    quantiles_result = fitted_forecaster.predict_quantiles(fh=fh)
    
    assert isinstance(quantiles_result, pd.DataFrame)

if __name__ == "__main__":
    pytest.main([__file__])