#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test forecasting module."""

__author__ = ["mloning"]

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from sktime.utils.validation.forecasting import check_fh

empty_input = (np.array([], dtype=int), [], pd.Index([], dtype=int))


@pytest.mark.parametrize("arg", empty_input)
def test_check_fh_empty_input(arg):
    """Test that fh validation throws an error with empty container."""
    with raises(ValueError, match="`fh` must not be empty"):
        check_fh(arg)
