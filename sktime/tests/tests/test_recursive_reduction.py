import pytest
import pandas as pd
from sktime.forecasting.compose import RecursiveReductionForecaster
from sktime.forecasting.naive import NaiveForecaster

@pytest.fixture
def sample_series():
    """Sample time series data with missing values."""
    return pd.Series([1.0, None, 3.0, None, 5.0])

def test_default_imputation(sample_series):
    base_forecaster = NaiveForecaster(strategy="mean")
    forecaster = RecursiveReductionForecaster(base_forecaster)
    forecaster.fit(sample_series)
    assert not forecaster._y.isnull().any()
