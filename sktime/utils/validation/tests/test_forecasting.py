#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_y

empty_input = (np.array([]), [], pd.Int64Index([]))
valid_numpy_input = np.array([3, 4, 5, 6])


@pytest.mark.parametrize("arg", empty_input)
def test_check_fh_empty_input(arg):
    with raises(ValueError):
        check_fh(arg)


# @pytest.mark.parametrize("arg", bad_input)
def test_check_y():
    isinstance(check_y(valid_numpy_input), pd.Series)
