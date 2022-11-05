# -*- coding: utf-8 -*-
"""Unit tests for sklearn classifiers."""

__author__ = ["MatthewMiddlehurst"]

from sklearn.utils.estimator_checks import parametrize_with_checks

from sktime.classification.sklearn import ContinuousIntervalTree, RotationForest


@parametrize_with_checks([RotationForest(n_estimators=3), ContinuousIntervalTree()])
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
