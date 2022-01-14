# -*- coding: utf-8 -*-
"""Catch22Classifier test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.feature_based import Catch22Classifier
from sktime.datasets import load_basic_motions, load_unit_test


def test_catch22_classifier_on_unit_test_data():
    """Test of Catch22Classifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train catch22 classifier
    c22c = Catch22Classifier(
        random_state=0,
        estimator=RandomForestClassifier(n_estimators=10),
        outlier_norm=True,
    )
    c22c.fit(X_train, y_train)

    # assert probabilities are the same
    probas = c22c.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, catch22_classifier_unit_test_probas, decimal=2
    )


def test_catch22_classifier_on_basic_motions():
    """Test of Catch22Classifier on basic motions."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train catch22 classifier
    c22c = Catch22Classifier(
        random_state=0, estimator=RandomForestClassifier(n_estimators=10)
    )
    c22c.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = c22c.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, catch22_classifier_basic_motions_probas, decimal=2
    )


catch22_classifier_unit_test_probas = np.array(
    [
        [
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.1,
            0.9,
        ],
        [
            0.8,
            0.2,
        ],
        [
            0.6,
            0.4,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.6,
            0.4,
        ],
        [
            0.2,
            0.8,
        ],
        [
            0.8,
            0.2,
        ],
        [
            0.8,
            0.2,
        ],
    ]
)
catch22_classifier_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.2,
            0.2,
            0.6,
        ],
        [
            0.0,
            0.9,
            0.0,
            0.1,
        ],
        [
            0.2,
            0.2,
            0.3,
            0.3,
        ],
        [
            0.1,
            0.5,
            0.4,
            0.0,
        ],
        [
            0.0,
            0.2,
            0.1,
            0.7,
        ],
        [
            0.0,
            0.0,
            0.4,
            0.6,
        ],
        [
            0.1,
            0.4,
            0.3,
            0.2,
        ],
        [
            0.2,
            0.0,
            0.6,
            0.2,
        ],
        [
            0.1,
            0.7,
            0.1,
            0.1,
        ],
        [
            0.0,
            0.9,
            0.0,
            0.1,
        ],
    ]
)
