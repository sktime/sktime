# -*- coding: utf-8 -*-
"""Tests for datetime functions."""

__author__ = ["xiaobenbenecho", "khrapovs"]

import datetime

import numpy as np
import pandas as pd

from sktime.datatypes import VectorizedDF
from sktime.utils._testing.hierarchical import _bottom_hier_datagen
from sktime.utils.datetime import _coerce_duration_to_int, _get_freq, infer_freq


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
