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
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train tde
    tde = TemporalDictionaryEnsemble(random_state=0)
    tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = tde.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, tde_gunpoint_probas)


def test_individual_tde_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train individual tde
    indiv_tde = IndividualTDE(random_state=0)
    indiv_tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = indiv_tde.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, individual_tde_gunpoint_probas)


tde_gunpoint_probas = np.array(
    [
        [0.08713241783457479, 0.9128675821654253, ],
        [0.2376248808557207, 0.7623751191442796, ],
        [0.7457478730539769, 0.2542521269460233, ],
        [0.625078546969323, 0.3749214530306776, ],
        [0.00915028065096904, 0.9908497193490308, ],
        [0.43692590108377183, 0.5630740989162286, ],
        [0.12405831891834646, 0.8759416810816537, ],
        [0.6518798319624394, 0.34812016803756146, ],
        [0.6410138736894136, 0.3589861263105872, ],
        [0.030515056306703855, 0.9694849436932959, ],
    ]
)
individual_tde_gunpoint_probas = np.array(
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
