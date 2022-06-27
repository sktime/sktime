# -*- coding: utf-8 -*-
"""Tests for datetime functions."""

__author__ = ["xiaobenbenecho", "khrapovs"]

import datetime

import numpy as np
import pandas as pd
import pytest

from sktime.utils.datetime import _coerce_duration_to_int, _get_freq, _shift


def test_get_freq():
    """Test whether get_freq runs without error."""
    x = pd.Series(
        index=pd.date_range(start="2017-01-01", periods=700, freq="W"),
        data=np.random.randn(700),
    )
    x1 = x.index
    x2 = x.resample("W").sum().index
    x3 = pd.Series(
        index=[
            datetime.datetime(2017, 1, 1) + datetime.timedelta(days=int(i))
            for i in np.arange(1, 100, 7)
        ],
        dtype=float,
    ).index
    x4 = [
        datetime.datetime(2017, 1, 1) + datetime.timedelta(days=int(i))
        for i in np.arange(1, 100, 7)
    ]
    assert _get_freq(x1) == "W"
    assert _get_freq(x2) == "W"
    assert _get_freq(x3) is None
    assert _get_freq(x4) is None


def test_coerce_duration_to_int() -> None:
    """Test _coerce_duration_to_int."""
    assert _coerce_duration_to_int(duration=0) == 0
    assert _coerce_duration_to_int(duration=3) == 3

    duration = pd.offsets.Minute(75)
    assert _coerce_duration_to_int(duration=duration, freq="25T") == 3

    duration = pd.Timedelta(minutes=75)
    assert _coerce_duration_to_int(duration=duration, freq="25T") == 3

    duration = pd.to_timedelta([75, 100], unit="m")
    pd.testing.assert_index_equal(
        _coerce_duration_to_int(duration=duration, freq="25T"),
        pd.Index([3, 4], dtype=int),
    )

    duration = pd.Index([pd.offsets.Minute(75), pd.offsets.Minute(100)])
    pd.testing.assert_index_equal(
        _coerce_duration_to_int(duration=duration, freq="25T"),
        pd.Index([3, 4], dtype=int),
    )


TIMEPOINTS = [
    pd.Period("2000", freq="D"),
    pd.Timestamp("2000-01-01", freq="D"),
    int(1),
    3,
]


@pytest.mark.parametrize("timepoint", TIMEPOINTS)
@pytest.mark.parametrize("by", [-3, -1, 0, 1, 3])
def test_shift(timepoint, by):
    """Test shifting of ForecastingHorizon."""
    ret = _shift(timepoint, by=by, freq="D")

    # check output type, pandas index types inherit from each other,
    # hence check for type equality here rather than using isinstance
    assert type(ret) is type(timepoint)

    # check if for a zero shift, input and output are the same
    if by == 0:
        assert timepoint == ret
