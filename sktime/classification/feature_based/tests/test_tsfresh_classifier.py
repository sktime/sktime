# -*- coding: utf-8 -*-
"""TSFreshClassifier test code."""
import numpy as np
import pytest
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.feature_based._tsfresh_classifier import TSFreshClassifier
from sktime.datasets import load_gunpoint, load_basic_motions, load_italy_power_demand


def test_tsfresh_classifier_on_gunpoint():
    """Test of TSFreshClassifier on gun point."""
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train TSFresh classifier
    rf = RandomForestClassifier(n_estimators=20)
    tsfc = TSFreshClassifier(
        random_state=0, default_fc_parameters="minimal", estimator=rf
    )
    tsfc.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = tsfc.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, tsfresh_classifier_gunpoint_probas)


@pytest.mark.parametrize("relevant_feature_extractor", [True, False])
def test_tsfresh_classifier_on_power_demand(relevant_feature_extractor):
    """Test of TSFreshClassifier on italy power demand."""
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train TSFresh classifier
    rf = RandomForestClassifier(n_estimators=20)
    tsfc = TSFreshClassifier(
        random_state=0,
        estimator=rf,
        relevant_feature_extractor=relevant_feature_extractor,
    )
    tsfc.fit(X_train, y_train)

    score = tsfc.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.9


def test_tsfresh_classifier_on_basic_motions():
    """Test of TSFreshClassifier on basic motions."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train TSFresh classifier
    rf = RandomForestClassifier(n_estimators=20)
    tsfc = TSFreshClassifier(
        random_state=0, default_fc_parameters="minimal", estimator=rf
    )
    tsfc.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = tsfc.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, tsfresh_classifier_basic_motions_probas)


tsfresh_classifier_gunpoint_probas = np.array(
    [
        [
            0.15,
            0.85,
        ],
        [
            0.3,
            0.7,
        ],
        [
            0.35,
            0.65,
        ],
        [
            0.25,
            0.75,
        ],
        [
            0.25,
            0.75,
        ],
        [
            0.45,
            0.55,
        ],
        [
            0.45,
            0.55,
        ],
        [
            0.1,
            0.9,
        ],
        [
            0.55,
            0.45,
        ],
        [
            0.45,
            0.55,
        ],
    ]
)
tsfresh_classifier_basic_motions_probas = np.array(
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
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.95,
            0.05,
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
            0.05,
            0.95,
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
            0.25,
            0.75,
        ],
        [
            0.95,
            0.05,
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
# if __name__ == "__main__":
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     rf = RandomForestClassifier(n_estimators=20)
#     tsfc_u = TSFreshClassifier(
#         random_state=0,
#         default_fc_parameters="minimal",
#         estimator=rf,
#     )
#
#     tsfc_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = tsfc_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     rf = RandomForestClassifier(n_estimators=20)
#     tsfc_m = TSFreshClassifier(
#         random_state=0,
#         default_fc_parameters="minimal",
#         estimator=rf,
#     )
#
#     tsfc_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = tsfc_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
