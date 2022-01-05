# -*- coding: utf-8 -*-
"""Retrieval utility for test scenarios."""

__author__ = ["fkiraly"]

__all__ = ["retrieve_scenarios"]


from inspect import isclass

from sktime.base import BaseObject
from sktime.registry import BASE_CLASS_LIST, BASE_CLASS_SCITYPE_LIST
from sktime.utils._testing.scenarios_forecasting import scenarios_forecasting

scenarios = dict()
scenarios["forecaster"] = scenarios_forecasting


def retrieve_scenarios(obj):
    """Retrieve test scenarios for obj, or by estimator scitype string.

    Exactly one of the arguments obj, estimator_type must be provided.

    Parameters
    ----------
    obj : class or object, or string.
        Which kind of estimator/object to retrieve scenarios for.
        If object, must be a class or object inheriting from BaseObject.
        If string, must be in registry.BASE_CLASS_REGISTER (first col)
            for instance 'classifier', 'regressor', 'transformer', 'forecaster'

    Returns
    -------
    scenarios : list of objects, instances of BaseScenario
    """
    if not isinstance(obj, (str, BaseObject)):
        raise TypeError("obj must be a str or inherit from BaseObject")
    if isinstance(obj, str) and obj not in BASE_CLASS_SCITYPE_LIST:
        raise ValueError(
            "if obj is a str, then obj must be a valid scitype string, "
            "see registry.BASE_CLASS_SCITYPE_LIST for valid scitype strings"
        )

    if not isinstance(obj, str):
        estimator_type = _scitype_from_class(obj)

    scenarios_for_type = scenarios.get(estimator_type)

    scenarios_for_type = [x() for x in scenarios_for_type if x.isapplicable(obj)]

    if scenarios_for_type is None:
        return []


def _scitype_from_class(obj):
    """Return scitype string given class or object."""
    if obj is None:
        raise ValueError("obj must not be None")
    if not isclass(obj):
        obj = type(obj)
    if not isinstance(obj, tuple(BASE_CLASS_LIST)):
        raise TypeError("obj must be instance of an sktime base class, or a base class")

    for i in range(len(BASE_CLASS_SCITYPE_LIST)):
        if isinstance(obj, BASE_CLASS_LIST[i]):
            return BASE_CLASS_SCITYPE_LIST[i]
