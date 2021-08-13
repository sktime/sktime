# -*- coding: utf-8 -*-
"""DrCIF test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.interval_based import DrCIF
from sktime.datasets import load_basic_motions, load_unit_test


def test_drcif_on_unit_test_data():
    """Test of DrCIF on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train DrCIF
    drcif = DrCIF(n_estimators=10, random_state=0, save_transformed_data=True)
    drcif.fit(X_train, y_train)

    # assert probabilities are the same
    probas = drcif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, drcif_unit_test_probas)

    # test train estimate
    train_probas = drcif._get_train_probs(X_train, y_train)
    train_preds = drcif.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.95

    assert accuracy_score(y_test, drcif.predict(X_test)) >= 0.95


def test_contracted_drcif_on_unit_test_data():
    """Test of contracted DrCIF on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)

    # train contracted DrCIF
    drcif = DrCIF(time_limit_in_minutes=0.05, random_state=0)
    drcif.fit(X_train, y_train)

    assert accuracy_score(y_test, drcif.predict(X_test)) >= 0.95


def test_drcif_on_basic_motions():
    """Test of DrCIF on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train DrCIF
    drcif = DrCIF(n_estimators=10, random_state=0)
    drcif.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = drcif.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, drcif_basic_motions_probas)

    assert accuracy_score(y_test, drcif.predict(X_test)) >= 0.95


drcif_unit_test_probas = np.array(
    [
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
            0.4,
            0.6,
        ],
        [
            0.05,
            0.95,
        ],
        [
            0.55,
            0.45,
        ],
        [
            0.1,
            0.9,
        ],
        [
            0.85,
            0.15,
        ],
        [
            0.45,
            0.55,
        ],
        [
            0.05,
            0.95,
        ],
    ]
)
drcif_basic_motions_probas = np.array(
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
            0.95,
            0.05,
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
            1.0,
            0.0,
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
            0.05,
            0.95,
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
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     drcif_u = DrCIF(n_estimators=10, random_state=0)
#
#     drcif_u.fit(X_train, y_train)
#     probas = drcif_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     drcif_m = DrCIF(n_estimators=10, random_state=0)
#
#     drcif_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = drcif_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
