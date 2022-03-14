# -*- coding: utf-8 -*-
"""TEASER test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import IsolationForest

from sktime.classification.early_classification._teaser import TEASER
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_unit_test
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy


def load_unit_data():
    """Load unit test data."""
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
    return X_test, X_train, indices, y_train


def test_teaser_on_unit_test_data():
    """Test of TEASER on unit test data."""
    X_test, X_train, indices, y_train = load_unit_data()

    # train probability threshold
    teaser = TEASER(
        random_state=0,
        classification_points=[6, 10, 16, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
    )
    teaser.fit(X_train, y_train)

    final_probas = np.zeros((10, 2))
    final_decisions = np.zeros(10)

    X_test = from_nested_to_3d_numpy(X_test)
    states = None
    for i in teaser.classification_points:
        X = X_test[indices, :, :i]
        probas, decisions, states = teaser.predict_proba(X, state_info=states)

        for n in range(10):
            if decisions[n] and final_decisions[n] == 0:
                final_probas[n] = probas[n]
                final_decisions[n] = i

    testing.assert_array_equal(final_probas, teaser_unit_test_probas)


def test_teaser_with_different_decision_maker():
    """Test of TEASER with different One-Class-Classifier."""
    X_test, X_train, indices, y_train = load_unit_data()

    # train probability threshold
    teaser = TEASER(
        random_state=0,
        classification_points=[6, 10, 16, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
        one_class_classifier=IsolationForest(n_estimators=5),
        one_class_param_grid={"bootstrap": [True, False]},
    )
    teaser.fit(X_train, y_train)

    final_probas = np.zeros((10, 2))
    final_decisions = np.zeros(10)

    X_test = from_nested_to_3d_numpy(X_test)
    states = None
    for i in teaser.classification_points:
        X = X_test[indices, :, :i]
        probas, decisions, states = teaser.predict_proba(X, state_info=states)

        for n in range(10):
            if decisions[n] and final_decisions[n] == 0:
                final_probas[n] = probas[n]
                final_decisions[n] = i

    testing.assert_array_equal(final_probas, teaser_if_unit_test_probas)


def test_teaser_near_classification_points():
    """Test of TEASER with incremental time stamps outside defined class points."""
    X_test, X_train, indices, y_train = load_unit_data()

    # train probability threshold
    teaser = TEASER(
        random_state=0,
        classification_points=[14, 16, 18, 20, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
        one_class_classifier=IsolationForest(n_estimators=5),
        one_class_param_grid={"bootstrap": [True, False]},
    )
    teaser.fit(X_train, y_train)

    final_probas = np.zeros((10, 2))
    final_decisions = np.zeros(10)

    # use test_points that are not within list above
    test_points = [15, 17, 19, 21, 24]
    X_test = from_nested_to_3d_numpy(X_test)
    states = None
    for i in test_points:
        X = X_test[indices, :, :i]
        probas, decisions, states = teaser.predict_proba(X, state_info=states)

        for n in range(10):
            if decisions[n] and final_decisions[n] == 0:
                final_probas[n] = probas[n]
                final_decisions[n] = i

    testing.assert_array_equal(final_probas, test_near_classification_points)


teaser_unit_test_probas = np.array(
    [
        [0.0, 1.0],
        [0.5, 0.5],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.7, 0.3],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.1, 0.9],
        [0.9, 0.1],
        [1.0, 0.0],
    ]
)

teaser_if_unit_test_probas = np.array(
    [
        [0.0, 1.0],
        [0.7, 0.3],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.7, 0.3],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.2, 0.8],
        [0.9, 0.1],
        [1.0, 0.0],
    ]
)

test_near_classification_points = np.array(
    [
        [0.1, 0.9],
        [0.7, 0.3],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.2, 0.8],
        [0.6, 0.4],
        [1.0, 0.0],
    ]
)
