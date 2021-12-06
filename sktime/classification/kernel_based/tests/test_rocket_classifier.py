# -*- coding: utf-8 -*-
"""RocketClassifier test code."""
import numpy as np
from numpy import testing

from sktime.classification.kernel_based import RocketClassifier
from sktime.datasets import load_basic_motions, load_unit_test


def test_rocket_on_unit_test_data():
    """Test of RocketClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train Rocket
    rocket = RocketClassifier(num_kernels=500, random_state=0)
    rocket.fit(X_train, y_train)

    # assert probabilities are the same
    probas = rocket.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, rocket_unit_test_probas, decimal=2)


def test_rocket_on_basic_motions():
    """Test of RocketClassifier on basic motions."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train Rocket
    rocket = RocketClassifier(num_kernels=500, random_state=0)
    rocket.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = rocket.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, rocket_basic_motions_probas, decimal=2)


rocket_unit_test_probas = np.array(
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
            1.0,
            0.0,
        ],
    ]
)
rocket_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
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
#     indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
#
#     rocket_u = RocketClassifier(num_kernels=500, random_state=0)
#
#     rocket_u.fit(X_train, y_train)
#     probas = rocket_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     rocket_m = RocketClassifier(num_kernels=500, random_state=0)
#
#     rocket_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = rocket_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
