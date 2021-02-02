# -*- coding: utf-8 -*-
import numpy as np
from numpy import testing

from sktime.classification.dictionary_based._tde import (
    TemporalDictionaryEnsemble,
    IndividualTDE,
)
from sktime.datasets import load_gunpoint, load_italy_power_demand, load_basic_motions


def test_tde_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train TDE
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=50,
        max_ensemble_size=10,
        randomly_selected_params=40,
        random_state=0,
    )
    tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = tde.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, tde_gunpoint_probas)


def test_individual_tde_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train IndividualTDE
    indiv_tde = IndividualTDE(random_state=0)
    indiv_tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = indiv_tde.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, individual_tde_gunpoint_probas)


def test_tde_on_power_demand():
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train TDE
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=50,
        max_ensemble_size=10,
        randomly_selected_params=40,
        random_state=0,
    )
    tde.fit(X_train, y_train)

    score = tde.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


def test_tde_on_basic_motions():
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(20)

    # train TDE
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=50,
        max_ensemble_size=10,
        randomly_selected_params=40,
        random_state=0,
    )
    tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = tde.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, tde_basic_motions_probas)


tde_gunpoint_probas = np.array(
    [
        [
            0.0,
            1.0000000000000002,
        ],
        [
            0.3291364535266975,
            0.6708635464733027,
        ],
        [
            0.5854317732366514,
            0.4145682267633488,
        ],
        [
            0.25629531970995384,
            0.7437046802900463,
        ],
        [
            0.0,
            1.0000000000000002,
        ],
        [
            0.4271588661832565,
            0.5728411338167437,
        ],
        [
            0.0854317732366513,
            0.9145682267633489,
        ],
        [
            0.5125906394199078,
            0.48740936058009243,
        ],
        [
            0.5125906394199078,
            0.48740936058009243,
        ],
        [
            0.0,
            1.0000000000000002,
        ],
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
tde_basic_motions_probas = np.array(
    [
        [
            0.8,
            0.2,
        ],
        [
            0.1,
            0.9,
        ],
        [
            0.7,
            0.3,
        ],
        [
            0.1,
            0.9,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.1,
            0.9,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.1,
            0.9,
        ],
        [
            0.1,
            0.9,
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
            0.0,
            1.0,
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
        [
            0.0,
            1.0,
        ],
        [
            0.2,
            0.8,
        ],
        [
            0.7,
            0.3,
        ],
        [
            0.9,
            0.1,
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
#     tde_u = TemporalDictionaryEnsemble(
#         n_parameter_samples=50,
#         max_ensemble_size=10,
#         randomly_selected_params=40,
#         random_state=0
#     )
#     indiv_tde = IndividualTDE(random_state=0)
#
#     tde_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = tde_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     indiv_tde.fit(X_train.iloc[indices], y_train[indices])
#     probas = indiv_tde.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(20)
#
#     tde_m = TemporalDictionaryEnsemble(
#         n_parameter_samples=50,
#         max_ensemble_size=10,
#         randomly_selected_params=40,
#         random_state=0
#     )
#
#     tde_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = tde_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
