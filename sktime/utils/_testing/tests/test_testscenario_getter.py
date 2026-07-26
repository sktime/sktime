#!/usr/bin/env python3 -u

__author__ = ["fkiraly"]
__all__ = []

import numpy as np
import pytest

from sktime.registry import get_base_class_lookup, get_obj_scitype_list
from sktime.utils._testing.scenarios import TestScenario
from sktime.utils._testing.scenarios_getter import retrieve_scenarios


@pytest.fixture(params=get_obj_scitype_list())
def estimator_class(request):
    lookup = get_base_class_lookup()
    return lookup[request.param]


def test_get_scenarios_for_class(estimator_class):
    """Test retrieval of scenarios by class."""
    scenarios = retrieve_scenarios(obj=estimator_class)

    assert isinstance(scenarios, list), "return of retrieve_scenarios is not a list"
    assert np.all(isinstance(x, TestScenario) for x in scenarios), (
        "return of retrieve_scenarios is not a list of scenarios"
    )

    # todo: remove once fully refactored to scenarios
    # assert len(scenarios) > 0


@pytest.mark.parametrize("scitype_string", get_obj_scitype_list())
def test_get_scenarios_for_string(scitype_string):
    """Test retrieval of scenarios by string."""
    scenarios = retrieve_scenarios(obj=scitype_string)

    assert isinstance(scenarios, list), "return of retrieve_scenarios is not a list"
    assert np.all(isinstance(x, TestScenario) for x in scenarios), (
        "return of retrieve_scenarios is not a list of scenarios"
    )

    # todo: remove once fully refactored to scenarios
    # assert len(scenarios) > 0


def test_get_scenarios_errors():
    """Test that errors are raised for bad input args."""
    with pytest.raises(TypeError):
        retrieve_scenarios()

    with pytest.raises(TypeError):
        retrieve_scenarios(obj=1)
