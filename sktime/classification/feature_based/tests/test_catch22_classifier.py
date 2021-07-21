# -*- coding: utf-8 -*-
"""Catch22Classifier test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.feature_based import Catch22Classifier
from sktime.datasets import load_gunpoint, load_basic_motions, load_italy_power_demand


def test_catch22_classifier_on_gunpoint():
    """Test of Catch22Classifier on gun point."""
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train catch22 classifier
    rf = RandomForestClassifier(n_estimators=20)
    c22c = Catch22Classifier(random_state=0, estimator=rf)
    c22c.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = c22c.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, catch22_classifier_gunpoint_probas)


def test_matrix_profile_classifier_on_power_demand():
    """Test of Catch22Classifier on italy power demand."""
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train catch22 classifier
    rf = RandomForestClassifier(n_estimators=20)
    c22c = Catch22Classifier(random_state=0, estimator=rf)
    c22c.fit(X_train, y_train)

    score = c22c.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.86


def test_catch22_classifier_on_basic_motions():
    """Test of Catch22Classifier on basic motions."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train catch22 classifier
    rf = RandomForestClassifier(n_estimators=20)
    c22c = Catch22Classifier(random_state=0, estimator=rf)
    c22c.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = c22c.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, catch22_classifier_basic_motions_probas)


catch22_classifier_gunpoint_probas = np.array(
    [
        [
            0.05,
            0.95,
        ],
        [
            0.5,
            0.5,
        ],
        [
            0.35,
            0.65,
        ],
        [
            0.2,
            0.8,
        ],
        [
            0.1,
            0.9,
        ],
        [
            0.95,
            0.05,
        ],
        [
            0.15,
            0.85,
        ],
        [
            0.7,
            0.3,
        ],
        [
            0.7,
            0.3,
        ],
        [
            0.1,
            0.9,
        ],
    ]
)
catch22_classifier_basic_motions_probas = np.array(
    [
        [
            1.0,
            0.0,
        ],
        [
            0.2,
            0.8,
        ],
        [
            0.7,
            0.3,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.95,
            0.05,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.2,
            0.8,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.05,
            0.95,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.95,
            0.05,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.05,
            0.95,
        ],
        [
            0.8,
            0.2,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.15,
            0.85,
        ],
        [
            0.3,
            0.7,
        ],
        [
            0.75,
            0.25,
        ],
        [
            1.0,
            0.0,
        ],
    ]
)


# def print_array(array):
#     print("[")
#     for sub_array in array:
#         print("[")
#         for value in sub_array:
#             print(value.astype(str), end="")
#             print(", ")
#         print("],")
#     print("]")
#
#
# if __name__ == "__main__":
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     rf = RandomForestClassifier(n_estimators=20)
#     c22c_u = Catch22Classifier(random_state=0, estimator=rf)
#
#     c22c_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = c22c_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     rf = RandomForestClassifier(n_estimators=20)
#     c22c_m = Catch22Classifier(random_state=0, estimator=rf)
#
#     c22c_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = c22c_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
