# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests MSTL functionality."""

__all__ = ["test-MSTL"]
__authors__ = ["krishna-t"]

import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from sktime.transformations.series.detrend.mstl import MSTL

def test_transform_returns_correct_components():
    """Tests whether expected components are returned when 
    *return_components* parameter is switched on."""
    # Load our default test dataset
    series = load_airline()
    series.index = series.index.to_timestamp()
    
    # Initialize the MSTL transformer with specific parameters
    transformer = MSTL(periods=[3,12], return_components=True)

    # Fit the transformer to the data
    transformer.fit(series)

    # Transform the data
    transformed = transformer.transform(series)

    # Check if the transformed data has the expected components
    assert 'transformed' in transformed.columns, "Transformed component missing!"
    assert 'trend' in transformed.columns, "Trend component missing!"
    assert 'resid' in transformed.columns, "Residual component missing!"
    assert 'seasonal_3' in transformed.columns, "Seasonal component missing!" 
    assert 'seasonal_12' in transformed.columns, "Seasonal component missing!"
