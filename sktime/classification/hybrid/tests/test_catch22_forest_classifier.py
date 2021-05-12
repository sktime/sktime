# -*- coding: utf-8 -*-
import numpy as np
from numpy import testing

from sktime.classification.hybrid._catch22_forest_classifier import (
    Catch22ForestClassifier,
)
from sktime.datasets import load_gunpoint, load_basic_motions


def test_catch22_forest_classifier_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train c22f
    c22f = Catch22ForestClassifier(random_state=0)
    c22f.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = c22f.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, catch22_forest_classifier_gunpoint_probas)


def test_catch22_forest_classifier_on_basic_motions():
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train c22f
    c22f = Catch22ForestClassifier(random_state=0)
    c22f.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = c22f.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, catch22_forest_classifier_basic_motions_probas)


catch22_forest_classifier_gunpoint_probas = np.array(
    [
        [
            0.075,
            0.925,
        ],
        [
            0.45,
            0.55,
        ],
        [
            0.35,
            0.65,
        ],
        [
            0.16,
            0.84,
        ],
        [
            0.11,
            0.89,
        ],
        [
            0.855,
            0.145,
        ],
        [
            0.08,
            0.92,
        ],
        [
            0.7,
            0.3,
        ],
        [
            0.65,
            0.35,
        ],
        [
            0.085,
            0.915,
        ],
    ]
)
catch22_forest_classifier_basic_motions_probas = np.array(
    [
        [
            1.0,
            0.0,
        ],
        [
            0.135,
            0.865,
        ],
        [
            0.845,
            0.155,
        ],
        [
            0.005,
            0.995,
        ],
        [
            0.99,
            0.01,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.185,
            0.815,
        ],
        [
            0.99,
            0.01,
        ],
        [
            0.015,
            0.985,
        ],
        [
            0.045,
            0.955,
        ],
        [
            0.015,
            0.985,
        ],
        [
            0.985,
            0.015,
        ],
        [
            0.01,
            0.99,
        ],
        [
            0.01,
            0.99,
        ],
        [
            0.875,
            0.125,
        ],
        [
            0.995,
            0.005,
        ],
        [
            0.055,
            0.945,
        ],
        [
            0.24,
            0.76,
        ],
        [
            0.88,
            0.12,
        ],
        [
            1.0,
            0.0,
        ],
    ]
)


# def print_array(array):
#     print('[')
#     for sub_array in array:
#         print('[')
#         for value in sub_array:
#             print(value.astype(str), end='')
#             print(', ')
#         print('],')
#     print(']')
#
# if __name__ == "__main__":
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     c22f_u = Catch22ForestClassifier(random_state=0)
#
#     c22f_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = c22f_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     c22f_m = Catch22ForestClassifier(random_state=0)
#
#     c22f_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = c22f_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
