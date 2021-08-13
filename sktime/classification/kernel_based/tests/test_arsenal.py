# -*- coding: utf-8 -*-
"""Arsenal test code."""
import numpy as np
from numpy import testing

from sktime.classification.kernel_based import Arsenal
from sktime.datasets import load_basic_motions, load_unit_test


def test_arsenal_on_unit_test_data():
    """Test of Arsenal on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train Arsenal
    arsenal = Arsenal(
        num_kernels=500, n_estimators=5, random_state=0, save_transformed_data=True
    )
    arsenal.fit(X_train, y_train)

    # assert probabilities are the same
    probas = arsenal.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, arsenal_unit_test_probas)

    score = arsenal.score(X_test, y_test)
    assert score >= 0.95

    # train_probas = arsenal._get_train_probs(X_train, y_train)
    # test train estimate


# test contracting on unittest data


def test_arsenal_on_basic_motions():
    """Test of Arsenal on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train Arsenal
    arsenal = Arsenal(num_kernels=500, n_estimators=5, random_state=0)
    arsenal.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = arsenal.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, arsenal_basic_motions_probas)

    score = arsenal.score(X_test, y_test)
    assert score >= 0.95


arsenal_unit_test_probas = np.array(
    [
        [
            -0.0,
            1.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
    ]
)
arsenal_basic_motions_probas = np.array(
    [
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            1.0,
            -0.0,
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
#     arsenal_u = Arsenal(num_kernels=500, n_estimators=5, random_state=0)
#
#     arsenal_u.fit(X_train, y_train)
#     probas = arsenal_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     arsenal_m = Arsenal(num_kernels=500, n_estimators=5, random_state=0)
#
#     arsenal_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = arsenal_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
