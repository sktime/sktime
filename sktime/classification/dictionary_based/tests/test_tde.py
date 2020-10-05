# -*- coding: utf-8 -*-
import numpy as np
from numpy import testing

from sktime.classification.dictionary_based._tde import (
    TemporalDictionaryEnsemble,
    IndividualTDE,
)
from sktime.datasets import load_gunpoint


def test_tde_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train tde
    tde = TemporalDictionaryEnsemble(random_state=0)
    tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = tde.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, tde_gunpoint_probas)


def test_individual_tde_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train individual tde
    indiv_tde = IndividualTDE(random_state=0)
    indiv_tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = indiv_tde.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, individual_tde_gunpoint_probas)


tde_gunpoint_probas = np.array([
    [0.05712499747948301, 0.942875002520516, ],
    [0.236868106385982, 0.7631318936140177, ],
    [0.6944192392743598, 0.30558076072564017, ],
    [0.6460051485760759, 0.3539948514239242, ],
    [0.012911768461946917, 0.987088231538052, ],
    [0.3794688766559799, 0.6205311233440205, ],
    [0.09196862460427883, 0.9080313753957204, ],
    [0.654406871937572, 0.3455931280624283, ],
    [0.5724329374441288, 0.4275670625558717, ],
    [0.026132720343596882, 0.9738672796564022, ],
])
individual_tde_gunpoint_probas = np.array([
    [0.0, 1.0, ],
    [0.0, 1.0, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
    [1.0, 0.0, ],
    [1.0, 0.0, ],
    [0.0, 1.0, ],
])


# def print_array(array):
#     print('[')
#     for sub_array in array:
#         print('[', end='')
#         for value in sub_array:
#             print(value.astype(str), end='')
#             print(', ', end='')
#         print('],')
#     print(']')
#
#
# if __name__ == "__main__":
#     X_train, y_train = load_gunpoint(split='train', return_X_y=True)
#     X_test, y_test = load_gunpoint(split='test', return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     tde = TemporalDictionaryEnsemble(random_state=0)
#     indiv_tde = IndividualTDE(random_state=0)
#
#     tde.fit(X_train.iloc[indices], y_train[indices])
#     probas = tde.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     indiv_tde.fit(X_train.iloc[indices], y_train[indices])
#     probas = indiv_tde.predict_proba(X_test.iloc[indices])
#     print_array(probas)
