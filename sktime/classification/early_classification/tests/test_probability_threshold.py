# -*- coding: utf-8 -*-
"""ProbabilityThresholdEarlyClassifier test code."""
import numpy as np
import pytest

from sktime.classification.early_classification import (
    ProbabilityThresholdEarlyClassifier,
)
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_unit_test
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy


def test_early_prob_threshold_near_classification_points():
    """Test of threshold with incremental time stamps outside defined class points."""
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train probability threshold
    teaser = ProbabilityThresholdEarlyClassifier(
        random_state=0,
        classification_points=[6, 10, 14, 18, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
    )
    teaser.fit(X_train, y_train)

    # use test_points that are not within list above
    test_points = [7, 11, 19, 20]

    X_test = from_nested_to_3d_numpy(X_test)
    X_test = X_test[indices]

    decisions = np.zeros(len(X_test), dtype=bool)
    for i in test_points:
        X_test = X_test[np.invert(decisions)]
        X = X_test[:, :, :i]

        if i == 20:
            with pytest.raises(ValueError):
                teaser.update_predict_proba(X)
        else:
            _, decisions = teaser.update_predict_proba(X)
