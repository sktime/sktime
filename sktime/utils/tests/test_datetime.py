# -*- coding: utf-8 -*-
"""Tests for datetime functions."""

__author__ = ["xiaobenbenecho", "khrapovs"]

import datetime

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import VectorizedDF
from sktime.utils._testing.hierarchical import _bottom_hier_datagen
from sktime.utils.datetime import _coerce_duration_to_int, _get_freq, _shift, infer_freq


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


def test_infer_freq() -> None:
    """Test frequency inference."""
    assert infer_freq(None) is None

    y = pd.Series(dtype=int)
    assert infer_freq(y) is None

    index = pd.date_range(start="2021-01-01", periods=1)
    y = pd.Series(index=index, dtype=int)
    assert infer_freq(y) == "D"

    index = pd.date_range(start="2021-01-01", periods=1, freq="M")
    y = pd.Series(index=index, dtype=int)
    assert infer_freq(y) == "M"

    y = pd.DataFrame({"a": 1}, index=pd.date_range(start="2021-01-01", periods=1))
    assert infer_freq(y) == "D"

    y = pd.DataFrame(
        {"a": 1}, index=pd.date_range(start="2021-01-01", periods=1, freq="M")
    )
    assert infer_freq(y) == "M"

    y = _bottom_hier_datagen(no_levels=2)
    y = VectorizedDF(X=y, iterate_as="Series", is_scitype="Hierarchical")
    assert infer_freq(y) == "M"


@pytest.mark.parametrize("x", [None, "a", [1]])
def test_shift_raises_attribute_error_with_wrong_x(x) -> None:
    """Test _shift raises AssertionError with wrong input x."""
    with pytest.raises(AssertionError, match=f"{type(x)}"):
        _shift(x)


@pytest.mark.parametrize(
    "by", [None, "a", [1], pd.Period("2021-01-01"), pd.Timedelta("1 day")]
)
def test_shift_raises_attribute_error_with_wrong_by(by) -> None:
    """Test _shift raises AssertionError with wrong input by."""
    with pytest.raises(AssertionError, match=f"{type(by)}"):
        _shift(pd.Period("2021-01-01"), by=by)


def test_shift_against_expectations() -> None:
    """Test _shift for equality with expectation."""
    assert _shift(1, by=2) == 3
    assert _shift(1, by=2, freq="D") == 3
    with pytest.raises(TypeError):
        _shift(pd.Timestamp("2021-01-01"))
    assert _shift(pd.Timestamp("2021-01-01"), freq="D") == pd.Timestamp("2021-01-02")
    assert _shift(pd.Timestamp("2021-01-01"), by=2, freq="D") == pd.Timestamp(
        "2021-01-03"
    )
    assert _shift(pd.Period("2021-01-15")) == pd.Period("2021-01-16")
    assert _shift(pd.Period("2021-01-15"), freq="D") == pd.Period("2021-01-16")
    assert _shift(pd.Period("2021-01-15"), by=2, freq="D") == pd.Period("2021-01-17")
    assert _shift(pd.Period("2021-01-15", freq="M")) == pd.Period("2021-02")
    assert _shift(pd.Period("2021-01-15", freq="M"), freq="D") == pd.Period("2021-02")
    assert _shift(pd.Period("2021-01-15", freq="M"), by=2) == pd.Period("2021-03")
    assert _shift(pd.Period("2021-01-15", freq="M"), by=2, freq="D") == pd.Period(
        "2021-03"
    )


@pytest.mark.parametrize(
    "timepoint", [pd.Period("2000", freq="D"), pd.Timestamp("2000-01-01"), int(1), 3]
)
@pytest.mark.parametrize("by", [-3, -1, 0, 1, 3])
def test_shift(timepoint, by):
    """Test _shift."""
    ret = _shift(timepoint, by=by, freq="D")

    # check output type, pandas index types inherit from each other,
    # hence check for type equality here rather than using isinstance
    assert type(ret) is type(timepoint)

    # check if for a zero shift, input and output are the same
    if by == 0:
        assert timepoint == ret
