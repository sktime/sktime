# -*- coding: utf-8 -*-
import numpy as np
from numpy import testing

from sktime.classification.interval_based._drcif import DrCIF
from sktime.datasets import load_gunpoint, load_italy_power_demand, load_basic_motions


def test_drcif_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train DrCIF
    drcif = DrCIF(n_estimators=20, random_state=0)
    drcif.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = drcif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, drcif_gunpoint_probas)


def test_drcif_on_power_demand():
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train DrCIF
    drcif = DrCIF(n_estimators=20, random_state=0)
    drcif.fit(X_train, y_train)

    score = drcif.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


def test_drcif_on_basic_motions():
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train DrCIF
    drcif = DrCIF(n_estimators=20, random_state=0)
    drcif.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = drcif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, drcif_basic_motions_probas)


drcif_gunpoint_probas = np.array(
    [
        [
            0.25,
            0.75,
        ],
        [
            0.55,
            0.45,
        ],
        [
            0.4,
            0.6,
        ],
        [
            0.4,
            0.6,
        ],
        [
            0.1,
            0.9,
        ],
        [
            0.45,
            0.55,
        ],
        [
            0.15,
            0.85,
        ],
        [
            0.55,
            0.45,
        ],
        [
            0.6,
            0.4,
        ],
        [
            0.2,
            0.8,
        ],
    ]
)
drcif_basic_motions_probas = np.array(
    [
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
            0.0,
            1.0,
        ],
        [
            0.95,
            0.05,
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
            0.9,
            0.1,
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
            0.05,
            0.95,
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
            0.0,
            1.0,
        ],
        [
            0.95,
            0.05,
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
            0.1,
            0.9,
        ],
        [
            0.95,
            0.05,
        ],
        [
            0.95,
            0.05,
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
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     drcif_u = DrCIF(
#         n_estimators=20,
#         random_state=0,
#     )
#
#     drcif_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = drcif_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     drcif_m = DrCIF(
#         n_estimators=20,
#         random_state=0,
#     )
#
#     drcif_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = drcif_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
