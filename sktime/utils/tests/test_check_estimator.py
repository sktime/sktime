"""Tests for check_estimator."""

__author__ = ["fkiraly"]

import pytest

from sktime.classification.dummy import DummyClassifier
from sktime.forecasting.dummy import ForecastKnownValues
from sktime.transformations.series.exponent import ExponentTransformer
from sktime.utils.estimator_checks import check_estimator

EXAMPLE_CLASSES = [DummyClassifier, ForecastKnownValues, ExponentTransformer]


@pytest.mark.parametrize("estimator_class", EXAMPLE_CLASSES)
def test_check_estimator_passed(estimator_class):
    """Test that check_estimator returns only passed tests for examples we know pass.

    Tests may be skipped if they are not applicable to the estimator,
    in this case the test is marked as "SKIP", and we test
    that less than 10% of tests are skipped.
    """
    estimator_instance = estimator_class.create_test_instance()

    result_class = check_estimator(estimator_class, verbose=False)

    # Check there are no failures.
    assert not any(x == "FAILED" for x in result_class.values())

    # Check less than 10% are skipped.
    skip_ratio = sum(list(x[:4] == "SKIP" for x in result_class.values()))
    skip_ratio = skip_ratio / len(result_class.values())
    assert skip_ratio < 0.1

    result_instance = check_estimator(estimator_instance, verbose=False)

    # Check there are no failures.
    assert not any(x == "FAILED" for x in result_instance.values())

    # Check less than 10% are skipped.
    skip_ratio = sum(list(x[:4] == "SKIP" for x in result_instance.values()))
    skip_ratio = skip_ratio / len(result_instance.values())
    assert skip_ratio < 0.1


@pytest.mark.parametrize("estimator_class", EXAMPLE_CLASSES)
def test_check_estimator_does_not_raise(estimator_class):
    """Test that check_estimator does not raise exceptions on examples we know pass."""
    estimator_instance = estimator_class.create_test_instance()

    check_estimator(estimator_class, raise_exceptions=True, verbose=False)

    check_estimator(estimator_instance, raise_exceptions=True, verbose=False)


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
