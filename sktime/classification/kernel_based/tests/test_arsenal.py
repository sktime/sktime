# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy import testing

from sktime.classification.kernel_based import Arsenal
from sktime.datasets import load_gunpoint, load_italy_power_demand, load_basic_motions


def test_arsenal_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train Arsenal
    arsenal = Arsenal(num_kernels=1000, n_estimators=10, random_state=0)
    arsenal.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = arsenal.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, arsenal_gunpoint_probas)


@pytest.mark.parametrize("n_jobs", [1, 8])
def test_arsenal_on_power_demand(n_jobs):
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train Arsenal
    arsenal = Arsenal(num_kernels=1000, n_estimators=10, random_state=0, n_jobs=n_jobs)
    arsenal.fit(X_train, y_train)

    score = arsenal.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


def test_arsenal_on_basic_motions():
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train Arsenal
    arsenal = Arsenal(num_kernels=1000, n_estimators=10, random_state=0)
    arsenal.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = arsenal.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, arsenal_basic_motions_probas)


arsenal_gunpoint_probas = np.array(
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
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     arsenal_u = Arsenal(num_kernels=1000, n_estimators=10, random_state=0)
#
#     arsenal_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = arsenal_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     arsenal_m = Arsenal(num_kernels=1000, n_estimators=10, random_state=0)
#
#     arsenal_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = arsenal_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
