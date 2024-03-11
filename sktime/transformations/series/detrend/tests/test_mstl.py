# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests MSTL functionality."""

__author__ = ["krishna-t"]

import pytest
import statsmodels as statsmodels

from sktime.datasets import load_airline
from sktime.transformations.series.detrend.mstl import MSTL


@pytest.mark.skipif(
    statsmodels.tsa.seasonal.MSTL is None, reason="MSTL is not installed"
)
def test_transform_returns_correct_components():
    """Tests whether expected components are returned when
    *return_components* parameter is switched on."""
    # Load our default test dataset
    series = load_airline()
    series.index = series.index.to_timestamp()

    # Initialize the MSTL transformer with specific parameters
    transformer = MSTL(periods=[3, 12], return_components=True)

    # Fit the transformer to the data
    transformer.fit(series)

    # Transform the data
    transformed = transformer.transform(series)

    # Check if the transformed data has the expected components
    assert "transformed" in transformed.columns, (
        "Test of MSTL.transform failed with return_components=True, "
        "returned DataFrame columns"
        " are missing 'transformed"
        " variable."
    )
    assert "trend" in transformed.columns, (
        "Test of MSTL.transform failed with return_components=True, "
        "returned DataFrame columns are "
        "missing 'trend' variable."
    )
    assert "resid" in transformed.columns, (
        "Test of MSTL.transform failed with return_components=True, "
        "returned DataFrame columns are "
        "missing 'resid' variable."
    )
    assert "seasonal_3" in transformed.columns, (
        "Test of MSTL.transform failed with return_components=True, "
        "returned DataFrame columns"
        " are missing 'seasonal_3 "
        "variable."
    )
    assert "seasonal_12" in transformed.columns, (
        "Test of MSTL.transform failed with return_components=True, "
        "returned DataFrame "
        "columns are missing "
        "'seasonal_12' variable."
    )
