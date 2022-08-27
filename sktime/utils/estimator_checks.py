# -*- coding: utf-8 -*-
"""Estimator checker for extension."""

__author__ = ["fkiraly"]
__all__ = ["check_estimator"]

from inspect import isclass


def check_estimator(
    estimator,
    return_exceptions=True,
    tests_to_run=None,
    fixtures_to_run=None,
    verbose=True,
    tests_to_exclude=None,
    fixtures_to_exclude=None,
):
    """Run all tests on one single estimator.

    Tests that are run on estimator:
        all tests in test_all_estimators
        all interface compatibility tests from the module of estimator's scitype
            for example, test_all_forecasters if estimator is a forecaster

    Parameters
    ----------
    estimator : estimator class or estimator instance
    return_exception : bool, optional, default=True
        whether to return exceptions/failures, or raise them
            if True: returns exceptions in results
            if False: raises exceptions as they occur
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

    Returns
    -------
    results : dict of results of the tests in self
        keys are test/fixture strings, identical as in pytest, e.g., test[fixture]
        entries are the string "PASSED" if the test passed,
            or the exception raised if the test did not pass
        returned only if all tests pass, or return_exceptions=True

    Raises
    ------
    if return_exception=False, raises any exception produced by the tests directly

    Examples
    --------
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>> from sktime.utils.estimator_checks import check_estimator
    >>> results = check_estimator(ThetaForecaster, tests_to_run="test_pred_int_tag")
    All tests PASSED!
    >>> check_estimator(
    ...    ThetaForecaster, fixtures_to_run="test_score[ThetaForecaster-y:1cols-fh=1]"
    ... )
    All tests PASSED!
    {'test_score[ThetaForecaster-y:1cols-fh=1]': 'PASSED'}
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
        return_exceptions=return_exceptions,
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
            return_exceptions=return_exceptions,
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
            return_exceptions=return_exceptions,
            tests_to_run=tests_to_run,
            fixtures_to_run=fixtures_to_run,
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
