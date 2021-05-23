# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._fh import DELEGATED_METHODS
from sktime.utils.datetime import _coerce_duration_to_int
from sktime.utils.datetime import _get_duration
from sktime.utils.datetime import _get_freq
from sktime.utils.datetime import _shift
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests._config import INDEX_TYPE_LOOKUP
from sktime.forecasting.tests._config import TEST_FHS
from sktime.forecasting.tests._config import VALID_INDEX_FH_COMBINATIONS
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.forecasting import _make_fh
from sktime.utils._testing.series import _make_index
from sktime.utils.validation.series import VALID_INDEX_TYPES


def _assert_index_equal(a, b):
    """Helper function to compare forecasting horizons"""
    assert isinstance(a, pd.Index)
    assert isinstance(b, pd.Index)
    assert a.equals(b)


@pytest.mark.parametrize(
    "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
)
@pytest.mark.parametrize("steps", TEST_FHS)
def test_fh(index_type, fh_type, is_relative, steps):
    # generate data
    y = make_forecasting_problem(index_type=index_type)
    assert isinstance(y.index, INDEX_TYPE_LOOKUP.get(index_type))

    # split data
    y_train, y_test = temporal_train_test_split(y, test_size=10)

    # choose cutoff point
    cutoff = y_train.index[-1]

    # generate fh
    fh = _make_fh(cutoff, steps, fh_type, is_relative)
    assert isinstance(fh.to_pandas(), INDEX_TYPE_LOOKUP.get(fh_type))

    # get expected outputs
    if isinstance(steps, int):
        steps = np.array([steps])
    fh_relative = pd.Int64Index(steps).sort_values()
    fh_absolute = y.index[np.where(y.index == cutoff)[0] + steps].sort_values()
    fh_indexer = fh_relative - 1
    fh_oos = fh.to_pandas()[fh_relative > 0]
    is_oos = len(fh_oos) == len(fh)
    fh_ins = fh.to_pandas()[fh_relative <= 0]
    is_ins = len(fh_ins) == len(fh)

    # check outputs
    # check relative representation
    _assert_index_equal(fh_absolute, fh.to_absolute(cutoff).to_pandas())
    assert not fh.to_absolute(cutoff).is_relative

    # check relative representation
    _assert_index_equal(fh_relative, fh.to_relative(cutoff).to_pandas())
    assert fh.to_relative(cutoff).is_relative

    # check index-like representation
    _assert_index_equal(fh_indexer, fh.to_indexer(cutoff))

    # check in-sample representation
    # we only compare the numpy array here because the expected solution is
    # formatted in a slightly different way than the generated solution
    np.testing.assert_array_equal(
        fh_ins.to_numpy(), fh.to_in_sample(cutoff).to_pandas()
    )
    assert fh.to_in_sample(cutoff).is_relative == is_relative
    assert fh.is_all_in_sample(cutoff) == is_ins

    # check out-of-sample representation
    np.testing.assert_array_equal(
        fh_oos.to_numpy(), fh.to_out_of_sample(cutoff).to_pandas()
    )
    assert fh.to_out_of_sample(cutoff).is_relative == is_relative
    assert fh.is_all_out_of_sample(cutoff) == is_oos


def test_fh_method_delegation():
    fh = ForecastingHorizon(1)
    for method in DELEGATED_METHODS:
        assert hasattr(fh, method)


BAD_INPUT_ARGS = (
    (1, 2),  # tuple
    "some_string",  # string
    0.1,  # float
    -0.1,  # negative float
    np.array([0.1, 2]),  # float in array
    None,
)


@pytest.mark.parametrize("arg", BAD_INPUT_ARGS)
def test_check_fh_values_bad_input_types(arg):
    with raises(TypeError):
        ForecastingHorizon(arg)


DUPLICATE_INPUT_ARGS = (
    np.array([1, 2, 2]),
    [3, 3, 1],
)


@pytest.mark.parametrize("arg", DUPLICATE_INPUT_ARGS)
def test_check_fh_values_duplicate_input_values(arg):
    with raises(ValueError):
        ForecastingHorizon(arg)


GOOD_INPUT_ARGS = (
    pd.Int64Index([1, 2, 3]),
    pd.period_range("2000-01-01", periods=3, freq="D"),
    pd.date_range("2000-01-01", periods=3, freq="M"),
    np.array([1, 2, 3]),
    [1, 2, 3],
    1,
)


@pytest.mark.parametrize("arg", GOOD_INPUT_ARGS)
def test_check_fh_values_input_conversion_to_pandas_index(arg):
    output = ForecastingHorizon(arg, is_relative=False).to_pandas()
    assert type(output) in VALID_INDEX_TYPES


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
    pd.TimedeltaIndex(range(0, 9, 3), unit="D", freq="3D"),
    pd.tseries.offsets.MonthEnd(3),
    pd.Index(pd.tseries.offsets.Day(day) for day in range(3)),
    # we also support pd.Timedelta, but it does not have freqstr so we
    # cannot automatically infer the unit during testing
    # pd.Timedelta(days=3, freq="D"),
]


@pytest.mark.parametrize("duration", DURATIONS)
def test_coerce_duration_to_int(duration):
    ret = _coerce_duration_to_int(duration, freq=_get_freq(duration))

    # check output type is always integer
    assert type(ret) in (pd.Int64Index, np.integer, int)

    # check result
    if isinstance(duration, pd.Index):
        np.testing.assert_array_equal(ret, range(3))

    if isinstance(duration, pd.tseries.offsets.BaseOffset):
        assert ret == 3


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
