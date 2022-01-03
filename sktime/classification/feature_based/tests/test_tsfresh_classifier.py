# -*- coding: utf-8 -*-
"""TSFreshClassifier test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.feature_based._tsfresh_classifier import TSFreshClassifier
from sktime.datasets import load_basic_motions, load_unit_test


def test_tsfresh_classifier_on_unit_test_data():
    """Test of TSFreshClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train TSFresh classifier
    tsfc = TSFreshClassifier(
        random_state=0,
        default_fc_parameters="minimal",
        relevant_feature_extractor=False,
        estimator=RandomForestClassifier(n_estimators=10),
    )
    tsfc.fit(X_train, y_train)

    # assert probabilities are the same
    probas = tsfc.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, tsfresh_classifier_unit_test_probas, decimal=2
    )


def test_tsfresh_classifier_on_basic_motions():
    """Test of TSFreshClassifier on basic motions."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train TSFresh classifier
    tsfc = TSFreshClassifier(
        random_state=0,
        default_fc_parameters="minimal",
        relevant_feature_extractor=False,
        estimator=RandomForestClassifier(n_estimators=10),
    )
    tsfc.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = tsfc.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, tsfresh_classifier_basic_motions_probas, decimal=2
    )


tsfresh_classifier_unit_test_probas = np.array(
    [
        [
            0.0,
            1.0,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.7,
            0.3,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.8,
            0.2,
        ],
        [
            0.9,
            0.1,
        ],
        [
            1.0,
            0.0,
        ],
        [
            1.0,
            0.0,
        ],
    ]
)
tsfresh_classifier_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.2,
            0.8,
        ],
        [
            0.4,
            0.2,
            0.1,
            0.3,
        ],
        [
            0.0,
            0.0,
            0.9,
            0.1,
        ],
        [
            0.0,
            0.9,
            0.0,
            0.1,
        ],
        [
            0.0,
            0.0,
            0.2,
            0.8,
        ],
        [
            0.0,
            0.0,
            0.3,
            0.7,
        ],
        [
            0.3,
            0.3,
            0.0,
            0.4,
        ],
        [
            0.0,
            0.0,
            0.9,
            0.1,
        ],
        [
            0.0,
            0.9,
            0.0,
            0.1,
        ],
        [
            0.1,
            0.9,
            0.0,
            0.0,
        ],
    ]
)
