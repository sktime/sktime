import pytest
import pandas as pd
from sktime.forecasting.base import BaseForecaster

# Define a minimal concrete class for testing
class MemoryAwareForecaster(BaseForecaster):
    def __init__(self, memory=None):
        super().__init__(memory=memory) 
    
    def _fit(self, y, X=None, fh=None):
        return self
    
    def _predict(self, fh, X=None):
        return None

def test_memory_limit_enforcement():
    """Test that memory limit raises MemoryError when exceeded."""
    # 1. Create Data (~2400 bytes)
    y = pd.Series([1, 2, 3] * 100)
    
    # 2. Set a limit smaller than the data (10 bytes)
    f = MemoryAwareForecaster(memory=10)
    
    # 3. Assert that fitting raises the error
    with pytest.raises(MemoryError, match="exceeds the memory limit"):
        f.fit(y)

def test_memory_limit_pass():
    """Test that fitting works when memory is sufficient."""
    y = pd.Series([1, 2, 3] * 100)
    
    # Set a generous limit (1 GB)
    f = MemoryAwareForecaster(memory=1024**3)
    
    # Should NOT raise an error
    f.fit(y)