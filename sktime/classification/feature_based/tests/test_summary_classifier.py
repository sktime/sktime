# -*- coding: utf-8 -*-
"""SummaryClassifier test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.feature_based import SummaryClassifier
from sktime.datasets import load_basic_motions, load_unit_test


def test_summary_classifier_on_unit_test_data():
    """Test of SummaryClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train summary stat classifier
    sc = SummaryClassifier(
        random_state=0, estimator=RandomForestClassifier(n_estimators=10)
    )
    sc.fit(X_train, y_train)

    # assert probabilities are the same
    probas = sc.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, summary_classifier_unit_test_probas, decimal=2
    )


def test_summary_classifier_on_basic_motions():
    """Test of SummaryClassifier on basic motions."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train summary stat classifier
    sc = SummaryClassifier(
        random_state=0, estimator=RandomForestClassifier(n_estimators=10)
    )
    sc.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = sc.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, summary_classifier_basic_motions_probas, decimal=2
    )


summary_classifier_unit_test_probas = np.array(
    [
        [
            0.1,
            0.9,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.4,
            0.6,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.8,
            0.2,
        ],
        [
            0.7,
            0.3,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.4,
            0.6,
        ],
        [
            0.6,
            0.4,
        ],
        [
            0.7,
            0.3,
        ],
    ]
)
summary_classifier_basic_motions_probas = np.array(
    [
        [
            0.1,
            0.0,
            0.2,
            0.7,
        ],
        [
            0.5,
            0.4,
            0.0,
            0.1,
        ],
        [
            0.1,
            0.1,
            0.8,
            0.0,
        ],
        [
            0.2,
            0.4,
            0.2,
            0.2,
        ],
        [
            0.1,
            0.0,
            0.0,
            0.9,
        ],
        [
            0.1,
            0.0,
            0.0,
            0.9,
        ],
        [
            0.5,
            0.3,
            0.0,
            0.2,
        ],
        [
            0.1,
            0.1,
            0.7,
            0.1,
        ],
        [
            0.1,
            0.4,
            0.3,
            0.2,
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
#     sc_u = SummaryClassifier(
#         random_state=0,
#         estimator=RandomForestClassifier(n_estimators=10),
#     )
#
#     sc_u.fit(X_train, y_train)
#     probas = sc_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     sc_m = SummaryClassifier(
#         random_state=0,
#         estimator=RandomForestClassifier(n_estimators=10),
#     )
#
#     sc_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = sc_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
