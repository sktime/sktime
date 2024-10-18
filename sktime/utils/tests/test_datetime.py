"""Tests for datetime functions."""

__author__ = ["xiaobenbenecho", "khrapovs"]

import datetime

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.datatypes import VectorizedDF
from sktime.datatypes._utilities import get_time_index
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _bottom_hier_datagen
from sktime.utils.datetime import (
    _coerce_duration_to_int,
    _get_freq,
    infer_freq,
    set_hier_freq,
)
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils", "sktime.datatypes"]),
    reason="Run if utils or datatypes module has changed.",
)
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils", "sktime.datatypes"]),
    reason="Run if utils or datatypes module has changed.",
)
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


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils", "sktime.datatypes"]),
    reason="Run if utils or datatypes module has changed.",
)
def test_infer_freq() -> None:
    """Test frequency inference."""
    assert infer_freq(None) is None

    y = pd.Series(dtype=int)
    assert infer_freq(y) is None

    index = pd.date_range(start="2021-01-01", periods=1)
    y = pd.Series(index=index, dtype=int)
    assert infer_freq(y) == "D"

    pandas2 = _check_soft_dependencies("pandas>=2.0.0", severity="none")
    if pandas2:
        m_freq = "ME"
    else:
        m_freq = "M"

    index = pd.date_range(start="2021-01-01", periods=1, freq=m_freq)
    y = pd.Series(index=index, dtype=int)
    assert infer_freq(y) in ["M", "ME"]

    y = pd.DataFrame({"a": 1}, index=pd.date_range(start="2021-01-01", periods=1))
    assert infer_freq(y) == "D"

    y = pd.DataFrame(
        {"a": 1}, index=pd.date_range(start="2021-01-01", periods=1, freq=m_freq)
    )
    assert infer_freq(y) in ["M", "ME"]

    y = _bottom_hier_datagen(no_levels=2)
    y = VectorizedDF(X=y, iterate_as="Series", is_scitype="Hierarchical")
    assert infer_freq(y) in ["M", "ME"]


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils", "sktime.datatypes"]),
    reason="Run if utils or datatypes module has changed.",
)
def test_set_freq_hier():
    """Test that setting frequency on a DatetimeIndex MultiIndex works."""
    # from pandas 2.1.0 on, freq is preserved correctly,
    # so in that case we artificially destroy the freq attribute
    pandas_210 = _check_soft_dependencies("pandas>=2.1.0", severity="none")

    y = load_airline()

    assert get_time_index(y).freq is not None

    # Convert to DatetimeIndex
    y_index = y.index.to_timestamp()

    assert get_time_index(y).freq is not None

    if pandas_210:
        y_index.freq = None

    # Create MultiIndex
    mi = pd.MultiIndex.from_product([[0], y_index], names=["instances", "timepoints"])
    y_group1 = pd.DataFrame(y.values, index=mi, columns=["y"])

    mi = pd.MultiIndex.from_product([[1], y_index], names=["instances", "timepoints"])
    y_group2 = pd.DataFrame(y.values, index=mi, columns=["y"])

    y_train_grp = pd.concat([y_group1, y_group2])

    assert get_time_index(y_train_grp).freq is None

    y_train_grp = set_hier_freq(y_train_grp)

    assert get_time_index(y_train_grp).freq is not None
