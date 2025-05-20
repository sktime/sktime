"""Unit tests for sklearn classifiers."""

__author__ = ["MatthewMiddlehurst"]

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktime.classification.sklearn import ContinuousIntervalTree, RotationForest
from sktime.tests.test_switch import run_test_for_class

ALL_SKLEARN_CLASSIFIERS = [RotationForest, ContinuousIntervalTree]

INSTANCES_TO_TEST = [
    RotationForest(n_estimators=3),
    RotationForest(n_estimators=3, base_estimator=LogisticRegression()),
    ContinuousIntervalTree(),
]


@pytest.mark.skipif(
    not run_test_for_class(ALL_SKLEARN_CLASSIFIERS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@parametrize_with_checks(INSTANCES_TO_TEST)
def test_sklearn_compatible_estimator(estimator, check):
    """Test that sklearn estimators adhere to sklearn conventions."""
    try:
        check(estimator)
    except AssertionError as error:
        # ContinuousIntervalTree can handle NaN values
        if not isinstance(
            estimator, ContinuousIntervalTree
        ) or "check for NaN and inf" not in str(error):
            raise error
