# -*- coding: utf-8 -*-
"""TDE test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.dictionary_based._tde import TemporalDictionaryEnsemble
from sktime.datasets import load_basic_motions, load_unit_test


def test_tde_on_unit_test_data():
    """Test of TDE on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train TDE
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=10,
        max_ensemble_size=5,
        randomly_selected_params=5,
        random_state=0,
    )
    tde.fit(X_train, y_train)

    # assert probabilities are the same
    probas = tde.predict_proba(X_test.iloc[indices]).round(6)
    testing.assert_array_equal(probas, tde_unit_test_probas)

    # test loocv train estimate
    train_probas = tde._get_train_probs(X_train, y_train)
    train_preds = tde.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75

    # test oob estimate
    train_probas = tde._get_train_probs(X_train, y_train, train_estimate_method="oob")
    train_preds = tde.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


# def test_contracted_tde_on_unit_test_data():
#     """Test of contracted TDE on unit test data."""
#     # load unit test data
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#
#     # train contracted TDE
#     tde = TemporalDictionaryEnsemble(
#         time_limit_in_minutes=0.25,
#         contract_max_n_parameter_samples=10,
#         max_ensemble_size=5,
#         randomly_selected_params=5,
#         random_state=0,
#     )
#     tde.fit(X_train, y_train)
#
#     assert len(tde.estimators_) > 1
#     assert accuracy_score(y_test, tde.predict(X_test)) >= 0.75


def test_tde_on_basic_motions():
    """Test of TDE on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train TDE
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=10,
        max_ensemble_size=5,
        randomly_selected_params=5,
        random_state=0,
    )
    tde.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = tde.predict_proba(X_test.iloc[indices]).round(6)
    testing.assert_array_equal(probas, tde_basic_motions_probas)


tde_unit_test_probas = np.array(
    [
        [
            0.0,
            1.0,
        ],
        [
            0.492418,
            0.507582,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.904333,
            0.095667,
        ],
        [
            0.801624,
            0.198376,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.705957,
            0.294043,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.801624,
            0.198376,
        ],
        [
            1.0,
            0.0,
        ],
    ]
)
tde_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            0.626119,
            0.373881,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.747762,
            0.0,
            0.0,
            0.252238,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            0.747762,
            0.252238,
            0.0,
        ],
        [
            0.0,
            0.747762,
            0.252238,
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
#     tde_u = TemporalDictionaryEnsemble(
#         n_parameter_samples=10,
#         max_ensemble_size=5,
#         randomly_selected_params=5,
#         random_state=0,
#     )
#
#     tde_u.fit(X_train, y_train)
#     probas = tde_u.predict_proba(X_test.iloc[indices]).round(6)
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     tde_m = TemporalDictionaryEnsemble(
#         n_parameter_samples=10,
#         max_ensemble_size=5,
#         randomly_selected_params=5,
#         random_state=0
#     )
#
#     tde_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = tde_m.predict_proba(X_test.iloc[indices]).round(6)
#     print_array(probas)
