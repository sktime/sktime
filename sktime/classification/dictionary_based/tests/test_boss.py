# -*- coding: utf-8 -*-
import numpy as np
from numpy import testing

from sktime.classification.dictionary_based import BOSSEnsemble, IndividualBOSS
from sktime.datasets import load_gunpoint, load_italy_power_demand


def test_boss_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train BOSS
    boss = BOSSEnsemble(random_state=0)
    boss.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = boss.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, boss_gunpoint_probas)


def test_individual_boss_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train IndividualBOSS
    indiv_boss = IndividualBOSS(random_state=0)
    indiv_boss.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = indiv_boss.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, individual_boss_gunpoint_probas)


def test_boss_on_power_demand():
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train BOSS
    boss = BOSSEnsemble(random_state=47)
    boss.fit(X_train, y_train)

    score = boss.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.9


boss_gunpoint_probas = np.array(
    [
        [
            0.0,
            1.0,
        ],
        [
            0.16,
            0.84,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.92,
            0.08,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.96,
            0.04,
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
individual_boss_gunpoint_probas = np.array(
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
#     boss = BOSSEnsemble(random_state=0)
#     indiv_boss = IndividualBOSS(random_state=0)
#
#     boss.fit(X_train.iloc[indices], y_train[indices])
#     probas = boss.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     indiv_boss.fit(X_train.iloc[indices], y_train[indices])
#     probas = indiv_boss.predict_proba(X_test.iloc[indices])
#     print_array(probas)
