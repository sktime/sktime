# -*- coding: utf-8 -*-
"""MatrixProfileClassifier test code."""
import numpy as np
from numpy import testing

from sktime.classification.feature_based import MatrixProfileClassifier
from sktime.datasets import load_gunpoint, load_italy_power_demand


def test_matrix_profile_classifier_on_gunpoint():
    """Test of MatrixProfileClassifier on gun point."""
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train matrix profile classifier
    mpc = MatrixProfileClassifier(random_state=0)
    mpc.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = mpc.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, matrix_profile_classifier_gunpoint_probas)


def test_matrix_profile_classifier_on_power_demand():
    """Test of MatrixProfileClassifier on italy power demand."""
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train TSFresh classifier
    mpc = MatrixProfileClassifier(random_state=0)
    mpc.fit(X_train, y_train)

    score = mpc.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.88


matrix_profile_classifier_gunpoint_probas = np.array(
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
#     print("[")
#     for sub_array in array:
#         print("[")
#         for value in sub_array:
#             print(value.astype(str), end="")
#             print(", ")
#         print("],")
#     print("]")
#
#
# if __name__ == "__main__":
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     mpc = MatrixProfileClassifier(random_state=0)
#
#     mpc.fit(X_train.iloc[indices], y_train[indices])
#     probas = mpc.predict_proba(X_test.iloc[indices])
#     print_array(probas)
