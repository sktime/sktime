# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.datasets import load_gunpoint, load_italy_power_demand


def test_y_proba_on_gunpoint():
    X, y = load_gunpoint(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    stsf = SupervisedTimeSeriesForest(random_state=42, n_estimators=20)
    stsf.fit(X_train, y_train)
    actual = stsf.predict_proba(X_test)
    np.testing.assert_array_equal(actual, expected)


def test_stsf_on_power_demand():
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train STSF
    stsf = SupervisedTimeSeriesForest(random_state=0, n_estimators=20)
    stsf.fit(X_train, y_train)

    score = stsf.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


# expected y_proba
expected = np.array(
    [
        [
            0.95,
            0.05,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.95,
            0.05,
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
            0.9,
            0.1,
        ],
        [
            0.05,
            0.95,
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
            0.2,
            0.8,
        ],
        [
            0.85,
            0.15,
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
            0.15,
            0.85,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.8,
            0.2,
        ],
        [
            1.0,
            0.0,
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
            0.0,
            1.0,
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
#     X, y = load_gunpoint(return_X_y=True)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.1, random_state=42
#     )
#     estimator = SupervisedTimeSeriesForest(random_state=42, n_estimators=20)
#     estimator.fit(X_train, y_train)
#     probas = estimator.predict_proba(X_test)
#     print_array(probas)
