#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.tests._config import INDEX_TYPE_LOOKUP
from sktime.utils._testing.forecasting import _make_index
from sktime.utils.datetime import _coerce_duration_to_int
from sktime.utils.datetime import _get_duration
from sktime.utils.datetime import _get_unit
from sktime.utils.datetime import _shift

TIMEPOINTS = [
    pd.Period("2000", freq="M"),
    pd.Timestamp("2000-01-01", freq="D"),
    np.int(1),
    3,
]


@pytest.mark.parametrize("timepoint", TIMEPOINTS)
@pytest.mark.parametrize("by", [-3, -1, 0, 1, 3])
def test_shift(timepoint, by):
    ret = _shift(timepoint, by=by)

    # check output type, pandas index types inherit from each other,
    # hence check for type equality here rather than using isinstance
    assert type(ret) is type(timepoint)

    # check if for a zero shift, input and output are the same
    if by == 0:
        assert timepoint == ret


DURATIONS = [
    pd.TimedeltaIndex(range(3), unit="D", freq="D"),
    pd.tseries.offsets.MonthEnd(3),
    pd.Index(pd.tseries.offsets.Day(day) for day in range(3)),
    # we also support pd.Timedelta, but it does not have freqstr so we
    # cannot automatically infer the unit during testing
    # pd.Timedelta(days=3, freq="D"),
]


@pytest.mark.parametrize("duration", DURATIONS)
def test_coerce_duration_to_int(duration):
    ret = _coerce_duration_to_int(duration, unit=_get_unit(duration))

    # check output type is always integer
    assert type(ret) in (pd.Int64Index, np.integer, int)

    # check result
    if isinstance(duration, pd.Index):
        np.testing.assert_array_equal(ret, range(3))

    if isinstance(duration, pd.tseries.offsets.BaseOffset):
        ret == 3


@pytest.mark.parametrize("n_timepoints", [3, 5])
@pytest.mark.parametrize("index_type", INDEX_TYPE_LOOKUP.keys())
def test_get_duration(n_timepoints, index_type):
    index = _make_index(n_timepoints, index_type)
    duration = _get_duration(index)
    # check output type is duration type
    assert isinstance(
        duration, (pd.Timedelta, pd.tseries.offsets.BaseOffset, int, np.integer)
    )

    # check integer output
    duration = _get_duration(index, coerce_to_int=True)
    assert isinstance(duration, (int, np.integer))
    assert duration == n_timepoints - 1
