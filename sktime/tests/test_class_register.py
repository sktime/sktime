# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Registry and dispatcher for test classes.

Module does not contain tests, only test utilities.
"""

__author__ = ["fkiraly"]

from inspect import isclass


def get_test_class_registry():
    """Return test class registry.

    Wrapped in a function to avoid circular imports.

    Returns
    -------
    testclass_dict : dict
        test class registry
        keys are scitypes, values are test classes TestAll[Scitype]
    """
    from sktime.registry._base_classes import _get_base_classes

    base_classes = _get_base_classes()

    testclass_dict = {}

    for base_class in base_classes:
        scitype_str = base_class.get_class_tag("scitype_name")
        test_class = base_class.get_test_class()

        if test_class is not None:
            testclass_dict[scitype_str] = test_class

    return testclass_dict


def get_test_classes_for_obj(obj):
    """Get all test classes relevant for an object or estimator.

    Parameters
    ----------
    obj : object or estimator, descendant of sktime BaseObject or BaseEstimator
        object or estimator for which to get test classes

    Returns
    -------
    test_classes : list of test classes
        list of test classes relevant for obj
        these are references to the actual classes, not strings
        if obj was not a descendant of BaseObject or BaseEstimator, returns empty list
    """
    from sktime.base import BaseEstimator, BaseObject
    from sktime.registry import scitype

    def is_object(obj):
        """Return whether obj is an estimator class or estimator object."""
        if isclass(obj):
            return issubclass(obj, BaseObject)
        else:
            return isinstance(obj, BaseObject)

    def is_estimator(obj):
        """Return whether obj is an estimator class or estimator object."""
        if isclass(obj):
            return issubclass(obj, BaseEstimator)
        else:
            return isinstance(obj, BaseEstimator)

    if not is_object(obj):
        return []

    testclass_dict = get_test_class_registry()

    # we always need to run "object" tests
    test_clss = [testclass_dict["object"]]

    if is_estimator(obj):
        test_clss += [testclass_dict["estimator"]]

    try:
        obj_scitypes = scitype(obj, force_single_scitype=False, coerce_to_list=True)
    except Exception:
        obj_scitypes = []

    for obj_scitype in obj_scitypes:
        if obj_scitype in testclass_dict:
            test_clss += [testclass_dict[obj_scitype]]

    return test_clss
