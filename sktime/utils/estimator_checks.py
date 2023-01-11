# -*- coding: utf-8 -*-
"""Estimator checker for extension."""

__author__ = ["fkiraly"]
__all__ = ["check_estimator"]

from inspect import isclass
from warnings import warn


# todo 0.17.0:
# * remove the return_exceptions arg
# * move the raise_exceptions arg to 2nd place
# * change its default to False, from None
# * update the docstring - remove return_exceptions
# * update the docstring - move raise_exceptions block to 2nd place
# * update the docstring - remove deprecation references
# * update the docstring - condition in return block, refer only to raise_exceptions
# * update the docstring - condition in raises block to refer only to raise_exceptions
# * remove the code block for input handling
# * remove import of warn
def check_estimator(
    estimator,
    return_exceptions=None,
    tests_to_run=None,
    fixtures_to_run=None,
    verbose=True,
    tests_to_exclude=None,
    fixtures_to_exclude=None,
    raise_exceptions=None,
):
    """Run all tests on one single estimator.

    Tests that are run on estimator:
        all tests in test_all_estimators
        all interface compatibility tests from the module of estimator's scitype
            for example, test_all_forecasters if estimator is a forecaster

    Parameters
    ----------
    estimator : estimator class or estimator instance
    return_exceptions : bool, optional, default=True
        whether to return exceptions/failures, or raise them
            if True: returns exceptions in returned `results` dict
            if False: raises exceptions as they occur
        deprecated since 0.15.1, and will be replaced by `raise_exceptions` in 0.17.0.
        Overridden to `False` if `raise_exceptions=True`.
        For safe deprecation, use `raise_exceptions` instead of `return_exceptions`.
    tests_to_run : str or list of str, optional. Default = run all tests.
        Names (test/function name string) of tests to run.
        sub-sets tests that are run to the tests given here.
    fixtures_to_run : str or list of str, optional. Default = run all tests.
        pytest test-fixture combination codes, which test-fixture combinations to run.
        sub-sets tests and fixtures to run to the list given here.
        If both tests_to_run and fixtures_to_run are provided, runs the *union*,
        i.e., all test-fixture combinations for tests in tests_to_run,
            plus all test-fixture combinations in fixtures_to_run.
    verbose : str, optional, default=True.
        whether to print out informative summary of tests run.
    tests_to_exclude : str or list of str, names of tests to exclude. default = None
        removes tests that should not be run, after subsetting via tests_to_run.
    fixtures_to_exclude : str or list of str, fixtures to exclude. default = None
        removes test-fixture combinations that should not be run.
        This is done after subsetting via fixtures_to_run.
    raise_exceptions : bool, optional, default=False
        whether to return exceptions/failures in the results dict, or raise them
            if False: returns exceptions in returned `results` dict
            if True: raises exceptions as they occur
        Overrides `return_exceptions` if used as a keyword argument.
        both `raise_exceptions=True` and `return_exceptions=True`.
        Will move to replace `return_exceptions` as 2nd arg in 0.17.0.

    Returns
    -------
    results : dict of results of the tests in self
        keys are test/fixture strings, identical as in pytest, e.g., test[fixture]
        entries are the string "PASSED" if the test passed,
            or the exception raised if the test did not pass
        returned only if all tests pass,
        or both return_exceptions=True and raise_exceptions=False

    Raises
    ------
    if return_exceptions=False, or raise_exceptions=True,
    raises any exception produced by the tests directly

    Examples
    --------
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> from sktime.utils.estimator_checks import check_estimator

    Running all tests for ExponentTransformer class,
    this uses all instances from get_test_params and compatible scenarios
    >>> results = check_estimator(ExponentTransformer)
    All tests PASSED!

    Running all tests for a specific ExponentTransformer
    this uses the instance that is passed and compatible scenarios
    >>> results = check_estimator(ExponentTransformer(42))
    All tests PASSED!

    Running specific test (all fixtures) for ExponentTransformer
    >>> results = check_estimator(ExponentTransformer, tests_to_run="test_clone")
    All tests PASSED!

    {'test_clone[ExponentTransformer-0]': 'PASSED',
    'test_clone[ExponentTransformer-1]': 'PASSED'}

    Running one specific test-fixture-combination for ExponentTransformer
    >>> check_estimator(
    ...    ExponentTransformer, fixtures_to_run="test_clone[ExponentTransformer-1]"
    ... )
    All tests PASSED!
    {'test_clone[ExponentTransformer-1]': 'PASSED'}
    """
    from sktime.base import BaseEstimator
    from sktime.classification.early_classification.tests.test_all_early_classifiers import (  # noqa E501
        TestAllEarlyClassifiers,
    )
    from sktime.classification.tests.test_all_classifiers import TestAllClassifiers
    from sktime.dists_kernels.tests.test_all_dist_kernels import (
        TestAllPairwiseTransformers,
        TestAllPanelTransformers,
    )
    from sktime.forecasting.tests.test_all_forecasters import TestAllForecasters
    from sktime.registry import scitype
    from sktime.regression.tests.test_all_regressors import TestAllRegressors
    from sktime.tests.test_all_estimators import TestAllEstimators, TestAllObjects
    from sktime.transformations.tests.test_all_transformers import TestAllTransformers

    # todo 0.17.0: remove this code block
    if return_exceptions is None and raise_exceptions is None:
        raise_exceptions = False

    if return_exceptions is not None and raise_exceptions is None:
        warn(
            "The return_exceptions argument of check_estimator has been deprecated "
            "since 0.15.1, and will be replaced by raise_exceptions in 0.17.0. "
            "For safe deprecation: use raise_exceptions argument instead of "
            "return_exceptions when using keywords. Avoid positional use, instead "
            "ensure to use keywords. When not using keywords, the "
            "default behaviour will not change."
        )
        raise_exceptions = not return_exceptions
    # end block to remove

    testclass_dict = dict()
    testclass_dict["classifier"] = TestAllClassifiers
    testclass_dict["early_classifier"] = TestAllEarlyClassifiers
    testclass_dict["forecaster"] = TestAllForecasters
    testclass_dict["regressor"] = TestAllRegressors
    testclass_dict["transformer"] = TestAllTransformers
    testclass_dict["transformer-pairwise"] = TestAllPairwiseTransformers
    testclass_dict["transformer-pairwise-panel"] = TestAllPanelTransformers

    results = TestAllObjects().run_tests(
        estimator=estimator,
        raise_exceptions=raise_exceptions,
        tests_to_run=tests_to_run,
        fixtures_to_run=fixtures_to_run,
        tests_to_exclude=tests_to_exclude,
        fixtures_to_exclude=fixtures_to_exclude,
    )

    def is_estimator(obj):
        """Return whether obj is an estimator class or estimator object."""
        if isclass(obj):
            return issubclass(obj, BaseEstimator)
        else:
            return isinstance(obj, BaseEstimator)

    if is_estimator(estimator):
        results_estimator = TestAllEstimators().run_tests(
            estimator=estimator,
            raise_exceptions=raise_exceptions,
            tests_to_run=tests_to_run,
            fixtures_to_run=fixtures_to_run,
            tests_to_exclude=tests_to_exclude,
            fixtures_to_exclude=fixtures_to_exclude,
        )
        results.update(results_estimator)

    try:
        scitype_of_estimator = scitype(estimator)
    except Exception:
        scitype_of_estimator = ""

    if scitype_of_estimator in testclass_dict.keys():
        results_scitype = testclass_dict[scitype_of_estimator]().run_tests(
            estimator=estimator,
            raise_exceptions=raise_exceptions,
            tests_to_run=tests_to_run,
            fixtures_to_run=fixtures_to_run,
            tests_to_exclude=tests_to_exclude,
            fixtures_to_exclude=fixtures_to_exclude,
        )
        results.update(results_scitype)

    failed_tests = [key for key in results.keys() if results[key] != "PASSED"]
    if len(failed_tests) > 0:
        msg = failed_tests
        msg = ["FAILED: " + x for x in msg]
        msg = "\n".join(msg)
    else:
        msg = "All tests PASSED!"

    if verbose:
        # printing is an intended feature, for console usage and interactive debugging
        print(msg)  # noqa T001

    return results
