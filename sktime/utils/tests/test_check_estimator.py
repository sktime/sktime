"""Tests for check_estimator."""

__author__ = ["fkiraly"]

import pytest

from sktime.classification.dummy import DummyClassifier
from sktime.forecasting.dummy import ForecastKnownValues
from sktime.tests.test_switch import run_test_for_class, run_test_module_changed
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.utils.estimator_checks import (
    _get_test_names_for_obj,
    check_estimator,
    parametrize_with_checks,
)

EXAMPLE_CLASSES = [DummyClassifier, ForecastKnownValues, ExponentTransformer]
EXAMPLE_INSTANCES = [cls.create_test_instance() for cls in EXAMPLE_CLASSES]


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils", "sktime.tests"])
    and not run_test_for_class(EXAMPLE_CLASSES),
    reason="Run if check_estimator or TestAll classes have changed.",
)
@pytest.mark.parametrize("estimator_class", EXAMPLE_CLASSES)
def test_check_estimator_passed(estimator_class):
    """Test that check_estimator returns only passed tests for examples we know pass.

    Tests may be skipped if they are not applicable to the estimator,
    in this case the test is marked as "SKIP", and we test
    that less than 10% of tests are skipped.
    """
    estimator_instance = estimator_class.create_test_instance()

    result_class = check_estimator(estimator_class, verbose=False)

    _check_none_failed_and_only_few_skipped(result_class)

    result_instance = check_estimator(estimator_instance, verbose=False)

    _check_none_failed_and_only_few_skipped(result_instance)


def _check_none_failed_and_only_few_skipped(result):
    """Check that no tests failed and only a few were skipped.

    Parameters
    ----------
    result : dict
        Dictionary of results from check_estimator.

    Raises
    ------
    AssertionError

        * If the input is not a dict.
        * If any tests failed, i.e., return is not "PASSED" or a skip-related string.
        * If more than 10% of tests were skipped, i.e., return is "SKIP".
    """
    assert isinstance(result, dict)
    assert all(isinstance(x, str) for x in result.values())

    # Check less than 10% are skipped.
    skip_ratio = sum([x[:4] == "SKIP" for x in result.values()])
    skip_ratio = skip_ratio / len(result.values())
    assert skip_ratio < 0.1


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils", "sktime.tests"])
    and not run_test_for_class(EXAMPLE_CLASSES),
    reason="Run if check_estimator or TestAll classes have changed.",
    # run_test_for_class will return True if TestAll has changed,
    # because test classes impacting EXAMPLE_CLASSES have changed
)
@pytest.mark.parametrize("estimator_class", EXAMPLE_CLASSES)
def test_check_estimator_does_not_raise(estimator_class):
    """Test that check_estimator does not raise exceptions on examples we know pass."""
    estimator_instance = estimator_class.create_test_instance()

    check_estimator(estimator_class, raise_exceptions=True, verbose=False)

    check_estimator(estimator_instance, raise_exceptions=True, verbose=False)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils", "sktime.tests"])
    and not run_test_for_class(ExponentTransformer),
    reason="Run if check_estimator or TestAll classes have changed.",
    # run_test_for_class will return True if TestAll has changed,
    # because it is a test class impacting ExponentTransformer
)
def test_check_estimator_subset_tests():
    """Test that subsetting by tests_to_run and tests_to_exclude works as intended."""
    tests_to_run = [
        "test_get_params",
        "test_set_params",
        "test_clone",
        "test_repr",
        "test_capability_inverse_tag_is_correct",
        "test_remember_data_tag_is_correct",
    ]
    tests_to_exclude = ["test_repr", "test_remember_data_tag_is_correct"]

    expected_tests = set(tests_to_run).difference(tests_to_exclude)

    results = check_estimator(
        ExponentTransformer,
        verbose=False,
        tests_to_run=tests_to_run,
        tests_to_exclude=tests_to_exclude,
    )
    results_tests = {x.split("[")[0] for x in results.keys()}

    assert results_tests == expected_tests


@pytest.mark.skipif(
    not run_test_for_class(_get_test_names_for_obj),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_get_test_names():
    """Test that get_test_names returns the correct test names."""
    expected_tests = [
        "test_get_params",
        "test_set_params",
        "test_clone",
        "test_repr",
        "test_capability_inverse_tag_is_correct",
        "test_remember_data_tag_is_correct",
    ]

    test_names = _get_test_names_for_obj(ExponentTransformer)

    assert set(expected_tests).issubset(test_names)


@pytest.mark.skipif(
    not run_test_for_class(parametrize_with_checks),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@parametrize_with_checks(EXAMPLE_CLASSES)
def test_parametrize_with_checks_objects(obj, test_name):
    """Test that parametrize_with_checks works as intended - classes."""
    check_estimator(obj, verbose=False, raise_exceptions=True, tests_to_run=test_name)


@pytest.mark.skipif(
    not run_test_for_class(parametrize_with_checks),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@parametrize_with_checks(EXAMPLE_INSTANCES, obj_varname="foo", check_varname="bar")
def test_parametrize_with_checks_instances(foo, bar):
    """Test that parametrize_with_checks works as intended - instances, var names."""
    check_estimator(foo, verbose=False, raise_exceptions=True, tests_to_run=bar)
