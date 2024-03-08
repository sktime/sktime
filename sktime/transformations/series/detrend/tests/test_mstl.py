import numpy as np
import pandas as pd
from sktime.transformations.series.detrend import MSTL

def test_mstl_returns_correct_components():
    # Create a time series dataset
    np.random.seed(42)
    data = np.random.randn(100)
    index = pd.date_range(start="2020-01-01", periods=100, freq="D")
    series = pd.Series(data, index=index)

    # Initialize the MSTL transformer with specific parameters
    transformer = MSTL(periods=7, return_components=True)

    # Fit the transformer to the data
    transformer.fit(series)

    # Transform the data
    transformed = transformer.transform(series)

    # Check if the transformed data has the expected components
    assert 'seasonal' in transformed.columns, "Seasonal component missing"
    assert 'trend' in transformed.columns, "Trend component missing"
    assert 'resid' in transformed.columns, "Residual component missing"

    # Optionally, check the inverse transform
    inverse_transformed = transformer.inverse_transform(transformed)
    assert np.allclose(series, inverse_transformed), "Inverse transform failed"

# Above tests MUST be modified because it's just a "template" kind of thing right now.
#  Also additional tests can be written to check for edge cases, incorrect parameters, etc.
