#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test series module."""

__author__ = ["benheid"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.utils.validation.series import check_equal_time_index, check_series


first_arrays = (np.random.random(1000),)
second_arrays = (np.random.random(1000),)


@pytest.mark.skipif(
    not run_test_for_class(check_equal_time_index),
    reason="Run if tested function has changed.",
)
@pytest.mark.parametrize("first_array", first_arrays)
@pytest.mark.parametrize("second_array", second_arrays)
def test_check_equal_time_index(first_array, second_array):
    """Test that fh validation throws an error with empty container."""
    check_equal_time_index(first_array, second_array)


@pytest.mark.skipif(
    not run_test_for_class(check_series),
    reason="Run if tested function has changed.",
)
def test_check_series_empty():
    """Test that check_series raises an error for empty pandas Series/DataFrame."""
    # Test with empty Series
    empty_series = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="Input time series is empty. Provide at least one observation."):
        check_series(empty_series)
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input time series is empty. Provide at least one observation."):
        check_series(empty_df)
    
    # Test that non-empty Series works fine
    non_empty_series = pd.Series([1, 2, 3])
    result = check_series(non_empty_series)
    assert result is non_empty_series
    
    # Test that non-empty DataFrame works fine
    non_empty_df = pd.DataFrame({"a": [1, 2, 3]})
    result = check_series(non_empty_df)
    assert result is non_empty_df
    
    # Test that empty Series is allowed when allow_empty=True
    result = check_series(empty_series, allow_empty=True)
    assert result is empty_series
    
    # Test that empty DataFrame is allowed when allow_empty=True
    result = check_series(empty_df, allow_empty=True)
    assert result is empty_df