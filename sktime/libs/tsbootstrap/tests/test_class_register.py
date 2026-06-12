# copyright: tsbootstrap developers, BSD-3-Clause License (see LICENSE file)
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
    from tsbootstrap.tests.test_all_bootstraps import TestAllBootstraps
    from tsbootstrap.tests.test_all_estimators import TestAllObjects

    testclass_dict = {}
    # every object in tsbootstrap inherits from BaseObject
    # "object" tests are run for all objects
    testclass_dict["object"] = TestAllObjects
    # more specific base classes
    # these inherit either from BaseEstimator or BaseObject,
    # so also imply estimator and object tests, or only object tests
    testclass_dict["bootstrap"] = TestAllBootstraps

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
    from skbase.base import BaseObject

    def is_object(obj):
        """Return whether obj is an estimator class or estimator object."""
        if isclass(obj):
            return issubclass(obj, BaseObject)
        else:
            return isinstance(obj, BaseObject)

    # warning: BaseEstimator does not inherit from BaseObject,
    # therefore we need to check both
    if not is_object(obj):
        return []

    testclass_dict = get_test_class_registry()

    # we always need to run "object" tests
    test_clss = [testclass_dict["object"]]

    try:
        obj_scitypes = obj.get_class_tag("object_type")
        if not isinstance(obj_scitypes, list):
            obj_scitypes = [obj_scitypes]
    except Exception:
        obj_scitypes = []

    for obj_scitype in obj_scitypes:
        if obj_scitype in testclass_dict:
            test_clss += [testclass_dict[obj_scitype]]

    return test_clss
