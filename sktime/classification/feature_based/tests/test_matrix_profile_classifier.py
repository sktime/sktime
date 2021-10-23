# -*- coding: utf-8 -*-
"""MatrixProfileClassifier test code."""
import numpy as np
from numpy import testing

from sktime.classification.feature_based import MatrixProfileClassifier
from sktime.datasets import load_unit_test


def test_matrix_profile_classifier_on_unit_test_data():
    """Test of MatrixProfileClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train matrix profile classifier
    mpc = MatrixProfileClassifier(random_state=0)
    mpc.fit(X_train, y_train)

    # assert probabilities are the same
    probas = mpc.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, matrix_profile_classifier_unit_test_probas, decimal=2
    )


matrix_profile_classifier_unit_test_probas = np.array(
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
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
#
#     mpc = MatrixProfileClassifier(random_state=0)
#
#     mpc.fit(X_train, y_train)
#     probas = mpc.predict_proba(X_test.iloc[indices])
#     print_array(probas)
