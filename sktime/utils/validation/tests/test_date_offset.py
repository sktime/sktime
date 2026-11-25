"""Tests for date offset validation."""

import warnings

import pandas as pd
import pytest

from sktime.utils.validation import (
    check_window_length,
    is_date_offset,
)


@pytest.mark.parametrize(
    "offset",
    [
        pd.offsets.Day(1),
        pd.offsets.Hour(2),
        pd.offsets.MonthEnd(1),
        pd.offsets.BusinessDay(1),
        pd.DateOffset(days=1),
    ],
)
def test_date_offset_validation(offset):
    """Valid offsets should be accepted without deprecation warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=r"isinstance\(obj, DateOffset\) is deprecated.*",
        )
        assert is_date_offset(offset)
        assert check_window_length(offset) == offset


@pytest.mark.parametrize(
    "value",
    [None, 1, 1.5, "D", pd.Timedelta(days=1)],
)
def test_is_date_offset_invalid_types(value):
    """Non-offset values should not be recognized as date offsets."""
    assert not is_date_offset(value)
