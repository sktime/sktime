import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from sktime.transformations.series.detrend import MSTL

def test_transform_returns_correct_components():
    # TODO: See if we need to create a fresh time series dataset?
    # np.random.seed(42)
    # data = np.random.randn(100)
    # index = pd.date_range(start="2020-01-01", periods=100, freq="D")
    # series = pd.Series(data, index=index)

    # Load our default test dataset
    series = load_airline()
    series.index = y.index.to_timestamp()
    
    # Initialize the MSTL transformer with specific parameters
    transformer = MSTL(periods=[3,12], return_components=True)

    # Fit the transformer to the data
    transformer.fit(series)

    # Transform the data
    transformed = transformer.transform(series)

    # Check if the transformed data has the expected components
    assert 'transformed' in transformed.columns, "Transformed component missing"
    assert 'trend' in transformed.columns, "Trend component missing"
    assert 'resid' in transformed.columns, "Residual component missing"
    # TODO: Specify "Seasonal_3" and "Seasonal_12" missing for our specific data or not?
    assert 'seasonal_3' in transformed.columns, "Seasonal component missing" 
    assert 'seasonal_12' in transformed.columns, "Seasonal component missing"
