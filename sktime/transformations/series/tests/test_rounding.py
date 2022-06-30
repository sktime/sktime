#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of Discretizer functionality."""

__author__ = ["AJarman"]
__all__ = []

import pytest

from sktime.transformations.series.rounding import Discretizer
from sktime.utils._testing.forecasting import make_forecasting_problem

y_ = make_forecasting_problem()


@pytest.mark.parametrize(
    "parameter_set",
    [
        "raise_ValueError_on_invalid_parameter_round_to_multiple",
        "raise_ValueError_on_invalid_parameter_round_to_list",
        "raise_ValueError_on_invalid_parameter_round_to_dp",
        "raise_ValueError_on_invalid_parameter_round_to_list_not_list",
    ],
)
def test_discretiser_raises_error(parameter_set):
    """Test that Discretizer raises ValueError on invalid parameters.

    Parameters
    ----------
    parameter_set : str
        string defining set of parameters in class's `get_test_params` method.
    """
    # get params
    testparams = Discretizer.get_test_params(parameter_set)

    with pytest.raises(ValueError):
        Discretizer(**testparams)


@pytest.mark.parametrize(
    "parameter_set",
    ["airline_round_to_multiple_and_dp", "airline_round_to_list_and_multiple"],
)
def test_discretiser_raises_warning(parameter_set):
    """Test the warning raised when multiple args passed.

    Parameters
    ----------
    parameter_set : str
        string defining set of parameters in class's `get_test_params` method.
    """
    # get params
    testparams = Discretizer.get_test_params(parameter_set)

    with pytest.warns(UserWarning):
        Discretizer(**testparams)
