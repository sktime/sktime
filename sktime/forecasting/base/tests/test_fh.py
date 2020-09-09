#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base._fh import DELEGATED_METHODS
from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tests import TEST_FHS
from sktime.utils._testing import make_forecasting_problem
from sktime.utils.validation import is_int

# currently supported combinations of
# data (y) index type, fh type, and is_relative option
SUPPORTED_COMBINATIONS = [
    ("int", "int", True),
    ("int", "int", False),
    ("range", "int", True),
    ("range", "int", False),
    ("period", "int", True),
    ("period", "period", False),
    ("datetime", "int", True),
    ("datetime", "datetime", False)
]

TYPE_LOOKUP = {
    "int": pd.Int64Index,
    "range": pd.RangeIndex,
    "datetime": pd.DatetimeIndex,
    "period": pd.PeriodIndex
}


def _make_index(y, index_class=None):
    """Helper function to make indices for testing"""
    n_timepoints = len(y)

    if index_class == "period":
        start = "2000-01"
        freq = "M"
        index = pd.period_range(start=start, periods=n_timepoints, freq=freq)

    elif index_class == "datetime":
        start = "2000-01"
        freq = "D"
        index = pd.date_range(start=start, periods=n_timepoints, freq=freq)

    elif index_class == "range":
        start = 3  # check non-zero based indices
        index = pd.RangeIndex(start=start, stop=start + n_timepoints)

    elif index_class == "int" or index_class is None:
        start = 3
        index = pd.Int64Index(np.arange(start, start + n_timepoints))

    else:
        raise ValueError(f"index_class: {index_class} is not supported")

    return index


def _make_fh(cutoff, steps, fh_type, is_relative):
    """Helper function to make forecasting horizons for testing"""
    fh_class = TYPE_LOOKUP.get(fh_type)

    if is_relative:
        return ForecastingHorizon(fh_class(steps), is_relative=is_relative)

    else:
        kwargs = {}

        if fh_type in ("int", "range"):
            values = cutoff + steps

        elif fh_type == "period":
            values = cutoff + steps
            kwargs = {"freq": cutoff.freq}

        elif fh_type == "datetime":
            values = cutoff + steps * cutoff.freq

        else:
            raise TypeError(f"Type of forecasting horizon not supported. "
                            f"Currently supported types are "
                            f"{TYPE_LOOKUP.values()}")

        return ForecastingHorizon(fh_class(values, **kwargs),
                                  is_relative=is_relative)


def _assert_index_equal(a, b):
    """Helper function to compare forecasting horizons"""
    assert isinstance(a, pd.Index)
    assert isinstance(b, pd.Index)
    assert a.equals(b)


@pytest.mark.parametrize("index_type, fh_type, is_relative",
                         SUPPORTED_COMBINATIONS)
@pytest.mark.parametrize("steps", TEST_FHS)
def test_fh(index_type, fh_type, is_relative, steps):
    # generate data
    y = make_forecasting_problem()

    # generate index
    y.index = _make_index(y, index_class=index_type)
    assert isinstance(y.index, TYPE_LOOKUP.get(index_type))

    # split data
    y_train, y_test = temporal_train_test_split(y, test_size=10)

    # choose cutoff point
    cutoff = y_train.index[-1]

    # generate fh
    if is_int(steps):
        steps = np.array([1], dtype=np.int)
    fh = _make_fh(cutoff, steps, fh_type, is_relative)
    assert isinstance(fh.to_pandas(), TYPE_LOOKUP.get(fh_type))

    # get expected outputs
    fh_relative = pd.Int64Index(steps).sort_values()
    fh_absolute = y.index[np.where(y.index == cutoff)[0] + steps].sort_values()
    fh_indexer = fh_relative - 1
    fh_oos = fh.to_pandas()[fh_relative > 0]
    fh_ins = fh.to_pandas()[fh_relative <= 0]

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
    np.testing.assert_array_equal(fh_ins.to_numpy(),
                                  fh.to_in_sample(cutoff).to_pandas())
    assert fh.to_in_sample(cutoff).is_relative == is_relative

    # check out-of-sample representation
    np.testing.assert_array_equal(fh_oos.to_numpy(),
                                  fh.to_out_of_sample(cutoff).to_pandas())
    assert fh.to_out_of_sample(cutoff).is_relative == is_relative


def test_fh_method_delegation():
    fh = _make_fh(10, np.arange(20), "int", True)
    for method in DELEGATED_METHODS:
        assert hasattr(fh, method)
