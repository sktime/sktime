"""Retrieval utility for test scenarios."""

# copied from sktime. Should be jointly refactored to scikit-base.

__author__ = ["fkiraly"]

__all__ = ["retrieve_scenarios"]


from inspect import isclass

from tsbootstrap.tests.scenarios.scenarios_bootstrap import scenarios_bootstrap

scenarios = {}
scenarios["bootstrap"] = scenarios_bootstrap


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
        subsets the returned objectss as follows:
            each key/value pair is statement in "and"/conjunction
                key is tag name to sub-set on
                value str or list of string are tag values
                condition is "key must be equal to value, or in set(value)"

    Returns
    -------
    scenarios : list of objects, instances of BaseScenario
    """
    # if class, get scitypes from inference; otherwise, str or list of str
    if not isinstance(obj, str):
        if isclass(obj):
            if hasattr(obj, "get_class_tag"):
                estimator_type = obj.get_class_tag("object_type", "object")
            else:
                estimator_type = "object"
        else:
            if hasattr(obj, "get_tag"):
                estimator_type = obj.get_tag("object_type", "object", False)
            else:
                estimator_type = "object"
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
        scenarios_for_type = [
            x for x in scenarios_for_type if x.is_applicable(obj)
        ]

    if filter_tags is not None:
        scenarios_for_type = [
            scen
            for scen in scenarios_for_type
            if _check_tag_cond(scen, filter_tags)
        ]

    return scenarios_for_type


def _check_tag_cond(obj, filter_tags=None):
    """Check whether object satisfies filter_tags condition.

    Parameters
    ----------
    obj: object inheriting from sktime BaseObject
    filter_tags: dict of (str or list of str), default=None
        subsets the returned objectss as follows:
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
