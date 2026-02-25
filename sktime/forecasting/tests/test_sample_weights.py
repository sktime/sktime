# Test file for sample_weight support
import pytest
import pandas as pd
import numpy as np
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import TrendForecaster

def test_base_raises_error_for_unsupported_weights():
    """Test that forecasters without the sample_weight capability raise an error."""
    # 1. Setup Data
    y = pd.Series([1, 2, 3], index=pd.period_range("2020-01-01", periods=3, freq="D"))
    weights = np.array([0.1, 0.5, 0.4])
    
    # 2. Use a forecaster that we know DOES NOT support weights
    forecaster = NaiveForecaster()
    
    # 3. Assert that it yells at us (NotImplementedError)
    msg = "does not support sample_weight"
    with pytest.raises(NotImplementedError, match=msg):
        forecaster.fit(y, sample_weight=weights)

def test_base_passes_weights_if_supported():
    """Test that weights are safely passed to _fit if the capability tag is True."""
    
    # 1. Define a Mock Forecaster that claims to support weights
    class MockWeightedForecaster(BaseForecaster):
        _tags = {
            "capability:sample_weight": True, # We explicitly turn this ON
            "requires-fh-in-fit": False,      # FIX: We say FH is not needed
            "scitype:y": "univariate"
        }
        
        def _fit(self, y, X=None, fh=None, sample_weight=None):
            # Capture the weights so we can check if they arrived
            self.captured_weight_ = sample_weight
            return self
            
        def _predict(self, fh, X=None):
            return pd.Series([1]*len(fh), index=fh)

    # 2. Setup Data
    y = pd.Series([1, 2, 3], index=pd.period_range("2020-01-01", periods=3, freq="D"))
    weights = np.array([0.1, 0.5, 0.4])
    
    # 3. Fit with weights
    forecaster = MockWeightedForecaster()
    forecaster.fit(y, sample_weight=weights)
    
    # 4. Verify the weights actually made it inside _fit
    assert hasattr(forecaster, "captured_weight_")
    assert forecaster.captured_weight_ is not None
    np.testing.assert_array_equal(forecaster.captured_weight_, weights)

def test_sample_weights_with_panel_data():
    """Test that sample_weight works with panel (multivariate) time series data."""
    # Panel data: multiple columns (multivariate)
    y_panel = pd.DataFrame(
        {
            'var1': [1.0, 2.0, 3.0, 4.0],
            'var2': [5.0, 6.0, 7.0, 8.0]
        },
        index=pd.period_range("2020-01-01", periods=4, freq="D")
    )
    
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    
    # Use TrendForecaster which supports sample_weight
    forecaster = TrendForecaster()
    forecaster.fit(y_panel, sample_weight=weights)
    
    # Verify forecaster is fitted
    assert forecaster._is_fitted


def test_sample_weights_with_univariate_long_series():
    """Test that sample_weight works with longer univariate time series."""
    # Create longer univariate series
    y_long = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        index=pd.period_range("2020-01-01", periods=10, freq="D")
    )
    
    # Create varied sample weights
    weights = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.05])
    
    # Use TrendForecaster which supports sample_weight
    forecaster = TrendForecaster()
    forecaster.fit(y_long, sample_weight=weights)
    
    # Verify forecaster is fitted
    assert forecaster._is_fitted