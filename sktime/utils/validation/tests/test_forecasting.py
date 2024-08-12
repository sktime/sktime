#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test forecasting module."""

__author__ = ["mloning"]

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from sktime.tests.test_switch import run_test_for_class
from sktime.utils.validation.forecasting import check_fh

empty_input = (np.array([], dtype=int), [], pd.Index([], dtype=int))


@pytest.mark.skipif(
    not run_test_for_class(check_fh),
    reason="Run if tested function has changed.",
)
@pytest.mark.parametrize("arg", empty_input)
def test_check_fh_empty_input(arg):
    """Test that fh validation throws an error with empty container."""
    with raises(ValueError, match="`fh` must not be empty"):
        check_fh(arg)
