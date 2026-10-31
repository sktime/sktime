"""Tests for the HampelFilter."""

__author__ = ["RobKuebler"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.outlier_detection import HampelFilter


@pytest.mark.skipif(
    not run_test_for_class(HampelFilter),
    reason="skip test only if softdeps are present and incrementally (if requested)",
)
def test_hampel_filter_series():
    """Verify that HampelFilter works as expected on single series."""
    # Create a sample time series with outliers
    data = pd.Series([1, 2, 3, 100, 5, 6, 7, 8, 9, 10])
    expected_result = data.replace(100, np.nan)

    hampel_filter = HampelFilter(window_length=3)
    result = hampel_filter.fit_transform(data)

    assert result.equals(expected_result)


@pytest.mark.skipif(
    not run_test_for_class(HampelFilter),
    reason="skip test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("window_length", [4, 5])
@pytest.mark.parametrize("outlier_position", [0, -2, -1])
@pytest.mark.parametrize("return_bool", [False, True])
def test_hampel_filter_single_window(window_length, outlier_position, return_bool):
    """Check every part of a series that fills exactly one window."""
    data = pd.Series(
        np.zeros(window_length),
        index=pd.date_range("2020-01-01", periods=window_length),
        name="signal",
    )
    data.iloc[outlier_position] = 100.0
    original = data.copy()
    expected = data.replace(100.0, np.nan)
    if return_bool:
        expected = expected.isna()

    result = HampelFilter(
        window_length=window_length, return_bool=return_bool
    ).fit_transform(data)

    pd.testing.assert_series_equal(data, original)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.skipif(
    not run_test_for_class(HampelFilter),
    reason="skip test only if softdeps are present and incrementally (if requested)",
)
def test_hampel_filter_panel():
    """Verify that HampelFilter works as expected on panel data."""
    # Create a sample time series with outliers
    data = pd.DataFrame(
        {
            "col1": [1, 2, 3, 100, 5, 6, 7, 8, 9, 10],
            "col2": [10, 9, 8, 7, 100, 5, 4, 3, 2, 1],
        },
        index=pd.MultiIndex.from_tuples(
            [(0, i) for i in range(10)], names=["group", "time"]
        ),
    )
    expected_result = data.replace(100, np.nan)

    hampel_filter = HampelFilter(window_length=3)
    result = hampel_filter.fit_transform(data)

    assert result.equals(expected_result)
