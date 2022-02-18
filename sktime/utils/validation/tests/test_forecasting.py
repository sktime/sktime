#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for Forecasting object."""

__author__ = ["Markus Löning"]

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from sktime.utils.validation.forecasting import check_fh

empty_input = (np.array([]), [], pd.Index([], dtype="int64"))


@pytest.mark.parametrize("arg", empty_input)
def test_check_fh_empty_input(arg):
    """Test that check_fh() raises ValueError for empty input."""
    with raises(ValueError):
        check_fh(arg)
