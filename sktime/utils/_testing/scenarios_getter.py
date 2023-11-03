"""Retrieval utility for test scenarios."""

__author__ = ["fkiraly"]

__all__ = ["retrieve_scenarios"]


from inspect import isclass

from sktime.base import BaseObject
from sktime.registry import BASE_CLASS_LIST, BASE_CLASS_SCITYPE_LIST, scitype
from sktime.utils._testing.scenarios_aligners import scenarios_aligners
from sktime.utils._testing.scenarios_classification import (
    scenarios_classification,
    scenarios_early_classification,
    scenarios_regression,
)
from sktime.utils._testing.scenarios_clustering import scenarios_clustering
from sktime.utils._testing.scenarios_forecasting import scenarios_forecasting
from sktime.utils._testing.scenarios_param_est import scenarios_param_est
from sktime.utils._testing.scenarios_transformers import scenarios_transformers
from sktime.utils._testing.scenarios_transformers_pairwise import (
    scenarios_transformers_pairwise,
    scenarios_transformers_pairwise_panel,
)

scenarios = dict()
scenarios["aligner"] = scenarios_aligners
scenarios["classifier"] = scenarios_classification
scenarios["early_classifier"] = scenarios_early_classification
scenarios["clusterer"] = scenarios_clustering
scenarios["forecaster"] = scenarios_forecasting
scenarios["param_est"] = scenarios_param_est
scenarios["regressor"] = scenarios_regression
scenarios["transformer"] = scenarios_transformers
scenarios["transformer-pairwise"] = scenarios_transformers_pairwise
scenarios["transformer-pairwise-panel"] = scenarios_transformers_pairwise_panel


def retrieve_scenarios(obj, filter_tags=None):
    """Retrieve test scenarios for obj, or by estimator scitype string.

    Exactly one of the arguments obj, estimator_type must be provided.

    Parameters
    ----------
    obj : class or object, or string, or list of str.
        Which kind of estimator/object to retrieve scenarios for.
        If object, must be a class or object inheriting from BaseObject.
        If string(s), must be in registry.BASE_CLASS_REGISTER (first col)
            for instance 'classifier', 'regressor', 'transformer', 'forecaster'
    filter_tags: dict of (str or list of str), default=None
        subsets the returned objects as follows:
            each key/value pair is statement in "and"/conjunction
                key is tag name to sub-set on
                value str or list of string are tag values
                condition is "key must be equal to value, or in set(value)"

    Returns
    -------
    scenarios : list of objects, instances of BaseScenario
    """
    if not isinstance(obj, (str, BaseObject)) and not issubclass(obj, BaseObject):
        raise TypeError("obj must be a str or inherit from BaseObject")
    if isinstance(obj, str) and obj not in BASE_CLASS_SCITYPE_LIST:
        raise ValueError(
            "if obj is a str, then obj must be a valid scitype string, "
            "see registry.BASE_CLASS_SCITYPE_LIST for valid scitype strings"
        )

    # if class, get scitypes from inference; otherwise, str or list of str
    if not isinstance(obj, str):
        estimator_type = scitype(obj)
    else:
        estimator_type = obj

    # coerce to list, ensure estimator_type is list of str
    if not isinstance(estimator_type, list):
        estimator_type = [estimator_type]

    # now loop through types and retrieve scenarios
    scenarios_for_type = []
    for est_type in estimator_type:
        scens = scenarios.get(est_type)
        if scens is not None:
            scenarios_for_type += scenarios.get(est_type)

    # instantiate all scenarios by calling constructor
    scenarios_for_type = [x() for x in scenarios_for_type]

    # if obj was an object, filter to applicable scenarios
    if not isinstance(obj, str) and not isinstance(obj, list):
        scenarios_for_type = [x for x in scenarios_for_type if x.is_applicable(obj)]

    if filter_tags is not None:
        scenarios_for_type = [
            scen for scen in scenarios_for_type if _check_tag_cond(scen, filter_tags)
        ]

    return scenarios_for_type


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


def _check_tag_cond(obj, filter_tags=None):
    """Check whether object satisfies filter_tags condition.

    Parameters
    ----------
    obj: object inheriting from sktime BaseObject
    filter_tags: dict of (str or list of str), default=None
        subsets the returned objects as follows:
            each key/value pair is statement in "and"/conjunction
                key is tag name to sub-set on
                value str or list of string are tag values
                condition is "key must be equal to value, or in set(value)"

    Returns
    -------
    cond_sat: bool, whether estimator satisfies condition in filter_tags
    """
    if not isinstance(filter_tags, dict):
        raise TypeError("filter_tags must be a dict")

    cond_sat = True

    for key, value in filter_tags.items():
        if not isinstance(value, list):
            value = [value]
        cond_sat = cond_sat and obj.get_class_tag(key) in set(value)

    return cond_sat
