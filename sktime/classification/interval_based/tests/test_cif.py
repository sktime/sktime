# -*- coding: utf-8 -*-
"""CanonicalIntervalForest test code."""
import numpy as np
from numpy import testing

from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.datasets import load_basic_motions, load_unit_test


def test_cif_on_unit_test_data():
    """Test of CanonicalIntervalForest on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train CIF
    cif = CanonicalIntervalForest(n_estimators=10, random_state=0)
    cif.fit(X_train, y_train)

    # assert probabilities are the same
    probas = cif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cif_unit_test_probas)


def test_cif_on_basic_motions():
    """Test of CanonicalIntervalForest on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train CIF
    cif = CanonicalIntervalForest(n_estimators=10, random_state=0)
    cif.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = cif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cif_basic_motions_probas)


cif_unit_test_probas = np.array(
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
            0.1,
            0.9,
        ],
        [
            1.0,
            0.0,
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
            1.0,
            0.0,
        ],
        [
            0.3,
            0.7,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.9,
            0.1,
        ],
    ]
)
cif_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.3,
            0.7,
        ],
        [
            0.5,
            0.4,
            0.1,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.8,
            0.2,
        ],
        [
            0.1,
            0.9,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.3,
            0.7,
        ],
        [
            0.0,
            0.0,
            0.4,
            0.6,
        ],
        [
            0.3,
            0.6,
            0.0,
            0.1,
        ],
        [
            0.0,
            0.1,
            0.5,
            0.4,
        ],
        [
            0.2,
            0.8,
            0.0,
            0.0,
        ],
        [
            0.2,
            0.7,
            0.0,
            0.1,
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
#
# if __name__ == "__main__":
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
#
#     cif_u = CanonicalIntervalForest(n_estimators=10, random_state=0)
#
#     cif_u.fit(X_train, y_train)
#     probas = cif_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     cif_m = CanonicalIntervalForest(n_estimators=10, random_state=0)
#
#     cif_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = cif_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
