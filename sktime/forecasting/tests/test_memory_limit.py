import pytest
import pandas as pd
from sktime.forecasting.naive import NaiveForecaster

def test_memory_limit_raises_error():
    """Test that memory limit raises MemoryError when exceeded."""
    # Create a tiny dataset (approx 8KB)
    y = pd.Series([1] * 1000)
    
    # Set a limit of 10 bytes (Guaranteed to fail)
    forecaster = NaiveForecaster(memory=10)
    
    with pytest.raises(MemoryError, match="exceeds the memory limit"):
        forecaster.fit(y)

def test_memory_limit_passes():
    """Test that fitting works when memory is sufficient."""
    y = pd.Series([1] * 1000)
    
    # Set a large limit (1GB)
    forecaster = NaiveForecaster(memory=10**9)
    forecaster.fit(y)