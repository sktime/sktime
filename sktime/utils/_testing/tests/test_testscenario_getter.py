#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["fkiraly"]
__all__ = []

import numpy as np
import pytest

from sktime.registry import BASE_CLASS_LIST, BASE_CLASS_SCITYPE_LIST

from sktime.utils._testing.scenarios import TestScenario
from sktime.utils._testing.scenarios_getter import retrieve_scenarios


@pytest.mark.parametrize("estimator_class", BASE_CLASS_LIST)
def test_get_scenarios_for_class(estimator_class):
    """Test retrieval of scenarios by class."""
    scenarios = retrieve_scenarios(obj=estimator_class)

    assert isinstance(scenarios, list), "return of retrieve_scenarios is not a list"
    assert np.all(isinstance(x, TestScenario) for x in scenarios), (
        "return of retrieve_scenarios is not a list of scenarios"
    )


@pytest.mark.parametrize("estimator_class", BASE_CLASS_SCITYPE_LIST)
def test_get_scenarios_for_string(scitype_string):
    """Test retrieval of scenarios by string."""
    scenarios = retrieve_scenarios(estimator_type=scitype_string)

    assert isinstance(scenarios, list), "return of retrieve_scenarios is not a list"
    assert np.all(isinstance(x, TestScenario) for x in scenarios), (
        "return of retrieve_scenarios is not a list of scenarios"
    )


def test_get_scenarios_errors():
    """Test that errors are raised for bad input args."""
    with pytest.raises(ValueError):
        retrieve_scenarios()

    with pytest.raises(ValueError):
        retrieve_scenarios(1, 2)

    with pytest.raises(ValueError):
        retrieve_scenarios(obj=1)

    with pytest.raises(ValueError):
        retrieve_scenarios(estimator_type=1)
