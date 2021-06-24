# -*- coding: utf-8 -*-
import numpy as np
from numpy import testing

from sktime.classification.interval_based._cif import CanonicalIntervalForest
from sktime.datasets import load_gunpoint, load_italy_power_demand, load_basic_motions


def test_cif_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train CIF
    cif = CanonicalIntervalForest(n_estimators=20, random_state=0)
    cif.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = cif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cif_gunpoint_probas)


def test_cif_on_power_demand():
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train CIF
    cif = CanonicalIntervalForest(n_estimators=20, random_state=0)
    cif.fit(X_train, y_train)

    score = cif.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


def test_cif_cit_on_power_demand():
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train CIF
    cif = CanonicalIntervalForest(n_estimators=20, base_estimator="CIT", random_state=0)
    cif.fit(X_train, y_train)

    score = cif.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


def test_cif_on_basic_motions():
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train CIF
    cif = CanonicalIntervalForest(n_estimators=20, random_state=0)
    cif.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = cif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cif_basic_motions_probas)


cif_gunpoint_probas = np.array(
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
            0.65,
            0.35,
        ],
        [
            0.3,
            0.7,
        ],
        [
            0.05,
            0.95,
        ],
        [
            0.75,
            0.25,
        ],
        [
            0.05,
            0.95,
        ],
        [
            0.65,
            0.35,
        ],
        [
            0.75,
            0.25,
        ],
        [
            0.15,
            0.85,
        ],
    ]
)
cif_basic_motions_probas = np.array(
    [
        [
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
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
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
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
            0.0,
            1.0,
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
#
# if __name__ == "__main__":
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     cif_u = CanonicalIntervalForest(n_estimators=20, random_state=0)
#
#     cif_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = cif_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     cif_m = CanonicalIntervalForest(n_estimators=20, random_state=0)
#
#     cif_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = cif_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
