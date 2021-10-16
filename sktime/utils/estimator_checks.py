# -*- coding: utf-8 -*-
"""Estimator checker for extendsion."""

__author__ = ["fkiraly"]
__all__ = ["check_estimator"]

from inspect import getfullargspec, getmembers, isfunction


def check_estimator(estimator, silent=False):
    """Run tests on estimator, manually.

    Parameters
    ----------
    estimator : sktime estimator
        estimator that is being tested
    silent : boolean, default=False
        if False, runs tests and raises errors
        if True, runs tests and returns failure/success list

    Raises
    ------
    any errors from the suite of estimator tests
        only if silent=False

    Returns
    -------
    result_list : list of (str, bool) pairs
        entries are (name of test function, whether test passed)
        only if silent=True
    """
    from sktime.tests import test_all_estimators

    # get all functions in the module test_allestimators that start with "test"
    all_funcs = getmembers(test_all_estimators, isfunction)
    test_funcs = [x for x in all_funcs if x[0].startswith("test")]

    # all tests have either a single argument estimator_class or estimator_instance
    # by checking how the argument is called, we can split the tests in class/object
    # note: if more complex fixture logic is introduced, this logic needs to adapt
    def firstarg(func):
        return getfullargspec(func[1]).args[0]

    estimator_class_tests = [x for x in test_funcs if firstarg(x) == "estimator_class"]
    estimator_obj_tests = [x for x in test_funcs if firstarg(x) == "estimator_instance"]

    # we will now run all tests
    # if silent=True, we also collect the results in result_list iteratively
    result_list = []

    # run all tests that operate on the class itself
    for class_test_name, class_test_func in estimator_class_tests:
        # if silent, we check whether the test raises exception and collect results
        if silent:
            try:
                class_test_func(estimator)
            except Exception:
                result_list.append((class_test_name, False))
            else:
                result_list.append((class_test_name, True))
        # if not silent, we just run the test and wait for errors to happen
        else:
            class_test_func(estimator)

    # run all tests that require an estimator instance
    #  these should be second, since create_test_instance is tested above
    for obj_test_name, obj_test_func in estimator_obj_tests:
        # we create a test instance, since the tests below require objects/instances
        # we have tested that create_test_instance works in the above batch of tests
        estimator_instance = estimator.create_test_instance()
        # if silent, we check whether the test raises exception and collect result
        if silent:
            try:
                obj_test_func(estimator_instance)
            except Exception:
                result_list.append((obj_test_name, False))
            else:
                result_list.append((obj_test_name, True))
        # if not silent, we just run the test and wait for errors to happen
        else:
            obj_test_func(estimator_instance)

    if silent:
        return result_list
    else:
        pass
