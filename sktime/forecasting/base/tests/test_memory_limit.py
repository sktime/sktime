import pytest
import pandas as pd
from sktime.forecasting.base import BaseForecaster

class MemoryAwareForecaster(BaseForecaster):
    # This tag tells sktime: "I don't need a forecasting horizon to train"
    _tags = {"requires-fh-in-fit": False}

    def __init__(self, memory=None):
        super().__init__(memory=memory)
    
    def _fit(self, y, X=None, fh=None):
        return self
    
    def _predict(self, fh, X=None):
        return None

def test_memory_limit_enforcement():
    """Test that memory limit raises MemoryError when exceeded."""
    y = pd.Series([1, 2, 3] * 100)
    # Limit is 10 bytes (too small)
    f = MemoryAwareForecaster(memory=10)
    
    with pytest.raises(MemoryError, match="exceeds the memory limit"):
        f.fit(y)

def test_memory_limit_pass():
    """Test that fitting works when memory is sufficient."""
    y = pd.Series([1, 2, 3] * 100)
    # Limit is 1 GB (plenty of space)
    f = MemoryAwareForecaster(memory=1024**3)
    
    # Should NOT raise an error
    f.fit(y)