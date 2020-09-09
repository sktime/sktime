#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_fh_values

bad_input_types = (
    (1, 2),  # tuple
    'some_string',  # string
    0.1,  # float
    -0.1,  # negative float
    np.array([0.1, 2]),  # float in array
    True,  # boolean
)

duplicate_input_values = (
    np.array([1, 2, 2]),  # duplicates
    [3, 3, 1],
)

empty_input = (
    np.array([]),  # empty
    [],
    pd.Int64Index([])
)

good_input_args = (
    pd.Int64Index([1, 2, 3]),
    pd.period_range("2000-01-01", periods=3, freq="D"),
    pd.date_range("2000-01-01", periods=3, freq="M"),
    np.array([1, 2, 3]),
    [1, 2, 3],
    1
)


@pytest.mark.parametrize("arg", empty_input)
def test_check_fh_empty_input(arg):
    with raises(ValueError):
        check_fh(arg)


@pytest.mark.parametrize("arg", bad_input_types)
def test_check_fh_values_bad_input_types(arg):
    with raises(TypeError):
        check_fh_values(arg)


@pytest.mark.parametrize("arg", duplicate_input_values)
def test_check_fh_values_duplicate_input_values(arg):
    with raises(ValueError):
        check_fh_values(arg)


@pytest.mark.parametrize("arg", good_input_args)
def test_check_fh_values_input_conversion_to_pandas_index(arg):
    output = check_fh_values(arg)
    assert isinstance(output, pd.Index)
