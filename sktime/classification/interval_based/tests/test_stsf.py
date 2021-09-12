# -*- coding: utf-8 -*-
"""STSF test code."""
import numpy as np
from numpy import testing

from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.datasets import load_gunpoint, load_italy_power_demand


def test_stsf_on_gunpoint():
    """Test of STSF on gun point."""
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    stsf = SupervisedTimeSeriesForest(n_estimators=20, random_state=0)
    stsf.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = stsf.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, stsf_gunpoint_probas)


def test_stsf_on_power_demand():
    """Test of STSF on italy power demand."""
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train STSF
    stsf = SupervisedTimeSeriesForest(random_state=0, n_estimators=20)
    stsf.fit(X_train, y_train)

    score = stsf.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


stsf_gunpoint_probas = np.array(
    [
        [
            0.15,
            0.85,
        ],
        [
            0.65,
            0.35,
        ],
        [
            0.7,
            0.3,
        ],
        [
            0.3,
            0.7,
        ],
        [
            0.2,
            0.8,
        ],
        [
            0.55,
            0.45,
        ],
        [
            0.35,
            0.65,
        ],
        [
            0.5,
            0.5,
        ],
        [
            0.4,
            0.6,
        ],
        [
            0.35,
            0.65,
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
#     stsf = SupervisedTimeSeriesForest(n_estimators=20, random_state=0)
#     stsf.fit(X_train.iloc[indices], y_train[indices])
#     probas = stsf.predict_proba(X_test.iloc[indices])
#     print_array(probas)
