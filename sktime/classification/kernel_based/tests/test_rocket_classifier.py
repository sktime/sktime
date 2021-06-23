# -*- coding: utf-8 -*-
import numpy as np
from numpy import testing
import pytest

from sktime.classification.kernel_based import ROCKETClassifier
from sktime.datasets import load_gunpoint, load_italy_power_demand, load_basic_motions


def test_rocket_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train ROCKET
    rocket = ROCKETClassifier(num_kernels=1000, random_state=0)
    rocket.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = rocket.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, rocket_gunpoint_probas)


@pytest.mark.parametrize("n_jobs", [1, 8])
def test_rocket_on_power_demand(n_jobs):
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train ROCKET
    rocket = ROCKETClassifier(num_kernels=1000, random_state=0, n_jobs=n_jobs)
    rocket.fit(X_train, y_train)

    score = rocket.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


def test_rocket_on_basic_motions():
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train ROCKET
    rocket = ROCKETClassifier(num_kernels=1000, random_state=0)
    rocket.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = rocket.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, rocket_basic_motions_probas)


rocket_gunpoint_probas = np.array(
    [
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
    ]
)
rocket_basic_motions_probas = np.array(
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
            1.0,
            0.0,
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
#     rocket_u = ROCKETClassifier(num_kernels=1000, random_state=0)
#
#     rocket_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = rocket_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     rocket_m = ROCKETClassifier(num_kernels=1000, random_state=0)
#
#     rocket_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = rocket_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
