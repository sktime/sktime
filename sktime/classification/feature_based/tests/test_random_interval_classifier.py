# -*- coding: utf-8 -*-
"""RandomIntervalClassifier test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.feature_based import RandomIntervalClassifier
from sktime.datasets import load_basic_motions, load_unit_test
from sktime.transformations.series.summarize import SummaryTransformer


def test_random_interval_classifier_on_unit_test_data():
    """Test of RandomIntervalClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train random interval classifier
    ric = RandomIntervalClassifier(
        random_state=0,
        n_intervals=5,
        interval_transformers=SummaryTransformer(
            summary_function=("mean", "std", "min", "max"),
            quantiles=(0.25, 0.5, 0.75),
        ),
        estimator=RandomForestClassifier(n_estimators=10),
    )
    ric.fit(X_train, y_train)

    # assert probabilities are the same
    probas = ric.predict_proba(X_test[indices])
    testing.assert_array_almost_equal(
        probas, random_interval_classifier_unit_test_probas, decimal=2
    )


def test_random_interval_classifier_on_basic_motions():
    """Test of RandomIntervalClassifier on basic motions."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train random interval classifier
    ric = RandomIntervalClassifier(
        random_state=0,
        n_intervals=5,
        interval_transformers=SummaryTransformer(
            summary_function=("mean", "std", "min", "max"),
            quantiles=(0.25, 0.5, 0.75),
        ),
        estimator=RandomForestClassifier(n_estimators=10),
    )
    ric.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = ric.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, random_interval_classifier_basic_motions_probas, decimal=2
    )


random_interval_classifier_unit_test_probas = np.array(
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
            0.9,
            0.1,
        ],
        [
            0.2,
            0.8,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.9,
            0.1,
        ],
    ]
)
random_interval_classifier_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.2,
            0.8,
        ],
        [
            0.2,
            0.3,
            0.1,
            0.4,
        ],
        [
            0.0,
            0.0,
            0.8,
            0.2,
        ],
        [
            0.2,
            0.6,
            0.0,
            0.2,
        ],
        [
            0.0,
            0.0,
            0.2,
            0.8,
        ],
        [
            0.0,
            0.1,
            0.5,
            0.4,
        ],
        [
            0.3,
            0.2,
            0.1,
            0.4,
        ],
        [
            0.0,
            0.0,
            0.9,
            0.1,
        ],
        [
            0.0,
            0.9,
            0.0,
            0.1,
        ],
        [
            0.2,
            0.8,
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
#     ric_u = RandomIntervalClassifier(
#         random_state=0,
#         n_intervals=5,
#         interval_transformers=                SummaryTransformer(
#                     summary_function=("mean", "std", "min", "max"),
#                     quantiles=(0.25, 0.5, 0.75),
#                 ),
#         estimator=RandomForestClassifier(n_estimators=10),
#     )
#
#     ric_u.fit(X_train, y_train)
#     probas = ric_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     ric_m = RandomIntervalClassifier(
#         random_state=0,
#         n_intervals=5,
#         interval_transformers=SummaryTransformer(
#             summary_function=("mean", "std", "min", "max"),
#             quantiles=(0.25, 0.5, 0.75),
#         ),
#         estimator=RandomForestClassifier(n_estimators=10),
#     )
#
#     ric_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = ric_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
