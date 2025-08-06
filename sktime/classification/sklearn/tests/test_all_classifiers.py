"""Unit tests for sklearn classifiers."""

__author__ = ["MatthewMiddlehurst"]

import platform

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktime.classification.sklearn import ContinuousIntervalTree, RotationForest
from sktime.tests.test_switch import run_test_module_changed

ALL_SKLEARN_CLASSIFIERS = [RotationForest, ContinuousIntervalTree]

INSTANCES_TO_TEST = [
    RotationForest(n_estimators=3),
    RotationForest(n_estimators=3, base_estimator=LogisticRegression()),
    ContinuousIntervalTree(),
]


def _handle_test_stepouts(estimator, error):
    """Handle specific test stepouts for different estimators and platforms."""
    # ContinuousIntervalTree can handle NaN values
    if isinstance(estimator, ContinuousIntervalTree) and "check for NaN and inf" in str(
        error
    ):
        return

    # Handle ARM architecture tolerance for RotationForest
    if isinstance(estimator, RotationForest) and platform.machine() == "aarch64":
        if "Arrays are not equal" in str(error) and "Mismatched elements: 1 " in str(
            error
        ):
            return

    # If no stepout conditions are met, raise the error
    raise error


@pytest.mark.skipif(
    not run_test_module_changed("sktime.classification.sklearn"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@parametrize_with_checks(INSTANCES_TO_TEST)
def test_sklearn_compatible_estimator(estimator, check):
    """Test that sklearn estimators adhere to sklearn conventions."""
    try:
        check(estimator)
    except AssertionError as error:
        _handle_test_stepouts(estimator, error)
