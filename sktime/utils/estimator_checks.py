"""Estimator checker for extension."""

__author__ = ["fkiraly"]
__all__ = ["check_estimator"]


def check_estimator(
    estimator,
    raise_exceptions=False,
    tests_to_run=None,
    fixtures_to_run=None,
    verbose=True,
    tests_to_exclude=None,
    fixtures_to_exclude=None,
):
    """Run all tests on one single estimator.

    This utility runs all tests from the unified API conformance suites
    applying to the estimator, including tests for the specific subtype
    and all supertypes.

    If ``estimator`` is an instance, tests are run on the specific instance
    and its class;
    if ``estimator`` is a class, tests are run on the class, and all instances
    constructed via its ``create_test_instances_and_names`` method.

    NOTE: individual tests not in the API conformance suites are not run.

    Example: if ``estimator`` is a forecaster, runs:

    * tests in ``TestAllObjects``, because every ``forecaster`` is an ``object``
    * tests in ``TestAllEstimators``, because every ``forecaster`` is an ``estimator``
    * tests in ``TestAllForecasters``

    In the example, we do not run a ``test_my_favourite_estimator`` test that is not
    in ``TestAll[Something]``, if the
    ``estimator`` is an instance of ``MyFavouriteEstimator``.

    Parameters
    ----------
    estimator : estimator class or estimator instance
        class or instance of the estimator to be tested

        * if class: tests are run on the class, and all instances
          constructed via its ``create_test_instances_and_names`` method
        * if instance: tests are run on the instance, and its class

    raise_exceptions : bool, optional, default=False
        whether to return exceptions/failures in the results dict, or raise them

        * if False: returns exceptions in returned ``results`` dict
        * if True: raises exceptions as they occur

    tests_to_run : str or list of str, optional. Default = run all tests.
        Names (test/function name string) of tests to run.
        sub-sets tests that are run to the tests given here.

    fixtures_to_run : str or list of str, optional. Default = run all tests.
        pytest test-fixture combination codes, which test-fixture combinations to run.
        sub-sets tests and fixtures to run to the list given here.
        If both tests_to_run and fixtures_to_run are provided, runs the *union*,
        i.e., all test-fixture combinations for tests in tests_to_run,
        plus all test-fixture combinations in fixtures_to_run.

    verbose : int or bool, optional, default=1.
        verbosity level for printouts from tests run.

        * 0 or False: no printout
        * 1 or True (default): print summary of test run, but no print from tests
        * 2: print all test output, including output from within the tests

    tests_to_exclude : str or list of str, names of tests to exclude. default = None
        removes tests that should not be run, after subsetting via tests_to_run.

    fixtures_to_exclude : str or list of str, fixtures to exclude. default = None
        removes test-fixture combinations that should not be run.
        This is done after subsetting via fixtures_to_run.

    Returns
    -------
    results : dict
        dictionary of results of the tests that were run

        keys are test/fixture strings, identical as in pytest,
        e.g., ``test[fixture]``;
        entries are the string ``"PASSED"`` if the test passed,
        or the exception raised if the test did not pass.

        ``results`` is returned only if all tests pass,
        or ``raise_exceptions=False``.

    Raises
    ------
    if ``raise_exceptions=True``,
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
    from sktime.utils.dependencies import (
        _check_estimator_deps,
        _check_soft_dependencies,
    )

    msg = (
        "check_estimator is a testing utility for developers, and "
        "requires pytest to be present "
        "in the python environment, but pytest was not found. "
        "pytest is a developer dependency and not included in the base "
        "sktime installation. Please run: `pip install pytest` to "
        "install the pytest package. "
        "To install sktime with all developer dependencies, run:"
        " `pip install sktime[dev]`"
    )
    _check_soft_dependencies("pytest", msg=msg)

    try:
        _check_estimator_deps(estimator)
    except ModuleNotFoundError as e:
        msg = (
            "check_estimator requires all dependencies of the tested object "
            "to be present in the python environment, "
            "but some were not found. "
            f"Details: {e}"
        )
        raise ModuleNotFoundError(msg) from e

    from sktime.tests.test_class_register import get_test_classes_for_obj

    test_clss_for_est = get_test_classes_for_obj(estimator)

    results = {}

    for test_cls in test_clss_for_est:
        test_cls_results = test_cls().run_tests(
            estimator=estimator,
            raise_exceptions=raise_exceptions,
            tests_to_run=tests_to_run,
            fixtures_to_run=fixtures_to_run,
            tests_to_exclude=tests_to_exclude,
            fixtures_to_exclude=fixtures_to_exclude,
            verbose=verbose if raise_exceptions else False,
        )
        results.update(test_cls_results)

    failed_tests = [key for key in results.keys() if results[key] != "PASSED"]
    if len(failed_tests) > 0:
        msg = failed_tests
        msg = ["FAILED: " + x for x in msg]
        msg = "\n".join(msg)
    else:
        msg = "All tests PASSED!"

    if int(verbose) > 0:
        # printing is an intended feature, for console usage and interactive debugging
        print(msg)  # noqa T001

    return results


def _get_test_names_from_class(test_cls):
    """Get all test names from a test class.

    Parameters
    ----------
    test_cls : class
        class of the test

    Returns
    -------
    test_names : list of str
        list of test names
    """
    test_names = [attr for attr in dir(test_cls) if attr.startswith("test")]

    return test_names


def _get_test_names_for_obj(obj):
    """Get all test names for an object.

    Parameters
    ----------
    obj : object
        object to get tests for

    Returns
    -------
    test_names : list of str
        list of test names
    """
    from sktime.tests.test_class_register import get_test_classes_for_obj

    test_clss_for_obj = get_test_classes_for_obj(obj)

    test_names = []
    for test_cls in test_clss_for_obj:
        test_names.extend(_get_test_names_from_class(test_cls))

    return test_names


def parametrize_with_checks(objs, obj_varname="obj", check_varname="test_name"):
    """Pytest specific decorator for parametrizing estimator checks.

    Designed for setting up API compliance checks in compatible 2nd and 3rd party
    libraries, using ``pytest.mark.parametrize``.

    Inspired by the ``sklearn`` utility of the same name.

    Parameters
    ----------
    objs : objects class or instance, or list thereof
        Objects to generate test names for.
    obj_varname : str, optional, default = 'obj'
        Name of the variable for objects to use in the parametrization.
    check_varname : str, optional, default = 'test_name'
        Name of the variable for test name strings to use in the parametrization.

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    See Also
    --------
    check_estimator : Check if estimator adheres to sktime APi contracts.

    Examples
    --------
    >>> from sktime.utils.estimator_checks import parametrize_with_checks
    >>> from sktime.forecasting.croston import Croston
    >>> from sktime.forecasting.naive import NaiveForecaster

    >>> @parametrize_with_checks(NaiveForecaster, obj_varname='estimator')
    ... def test_sktime_compatible_estimator(estimator, test_name):
    ...     check_estimator(estimator, tests_to_run=test_name, raise_exceptions=True)

    >>> @parametrize_with_checks([NaiveForecaster, Croston])
    ... def test_sktime_compatible_estimators(obj, test_name):
    ...     check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)
    """
    import pytest

    if not isinstance(objs, list):
        objs = [objs]

    test_names = []
    for obj in objs:
        tests_for_obj = _get_test_names_for_obj(obj)
        test_names.extend([(obj, test) for test in tests_for_obj])

    var_str = f"{obj_varname}, {check_varname}"
    return pytest.mark.parametrize(var_str, test_names)
