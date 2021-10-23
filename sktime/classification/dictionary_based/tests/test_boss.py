# -*- coding: utf-8 -*-
"""BOSS test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.dictionary_based import BOSSEnsemble, IndividualBOSS
from sktime.datasets import load_unit_test


def test_boss_on_unit_test_data():
    """Test of BOSS on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train BOSS
    boss = BOSSEnsemble(
        max_ensemble_size=5, random_state=0, save_train_predictions=True
    )
    boss.fit(X_train, y_train)

    # assert probabilities are the same
    probas = boss.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, boss_unit_test_probas, decimal=2)

    # test train estimate
    train_probas = boss._get_train_probs(X_train, y_train)
    train_preds = boss.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


def test_individual_boss_on_unit_test():
    """Test of IndividualBOSS on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train IndividualBOSS
    indiv_boss = IndividualBOSS(random_state=0)
    indiv_boss.fit(X_train, y_train)

    # assert probabilities are the same
    probas = indiv_boss.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, individual_boss_unit_test_probas, decimal=2
    )


boss_unit_test_probas = np.array(
    [
        [
            0.4,
            0.6,
        ],
        [
            0.4,
            0.6,
        ],
        [
            0.2,
            0.8,
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
            0.8,
            0.2,
        ],
        [
            0.8,
            0.2,
        ],
        [
            0.2,
            0.8,
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
individual_boss_unit_test_probas = np.array(
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
#     boss = BOSSEnsemble(max_ensemble_size=5, random_state=0)
#     indiv_boss = IndividualBOSS(random_state=0)
#
#     boss.fit(X_train, y_train)
#     probas = boss.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     indiv_boss.fit(X_train, y_train)
#     probas = indiv_boss.predict_proba(X_test.iloc[indices])
#     print_array(probas)
