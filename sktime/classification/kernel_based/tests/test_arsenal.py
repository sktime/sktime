# -*- coding: utf-8 -*-
"""Arsenal test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.kernel_based import Arsenal
from sktime.datasets import load_basic_motions, load_unit_test


def test_arsenal_on_unit_test_data():
    """Test of Arsenal on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train Arsenal
    arsenal = Arsenal(
        num_kernels=200, n_estimators=5, random_state=0, save_transformed_data=True
    )
    arsenal.fit(X_train, y_train)

    # assert probabilities are the same
    probas = arsenal.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, arsenal_unit_test_probas, decimal=2)

    # test train estimate
    train_probas = arsenal._get_train_probs(X_train, y_train)
    train_preds = arsenal.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


# def test_contracted_arsenal_on_unit_test_data():
#     """Test of contracted Arsenal on unit test data."""
#     # load unit test data
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#
#     # train contracted Arsenal
#     arsenal = Arsenal(
#         time_limit_in_minutes=0.25,
#         contract_max_n_estimators=5,
#         num_kernels=200,
#         random_state=0,
#     )
#     arsenal.fit(X_train, y_train)
#
#     assert len(arsenal.estimators_) > 1
#     assert accuracy_score(y_test, arsenal.predict(X_test)) >= 0.75


def test_arsenal_on_basic_motions():
    """Test of Arsenal on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train Arsenal
    arsenal = Arsenal(num_kernels=200, n_estimators=5, random_state=0)
    arsenal.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = arsenal.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, arsenal_basic_motions_probas, decimal=2)


arsenal_unit_test_probas = np.array(
    [
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
        ],
        [
            1.0,
            -0.0,
        ],
    ]
)
arsenal_basic_motions_probas = np.array(
    [
        [
            -0.0,
            -0.0,
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
            -0.0,
            -0.0,
        ],
        [
            -0.0,
            -0.0,
            1.0,
            -0.0,
        ],
        [
            -0.0,
            0.62674723,
            0.37325277,
            -0.0,
        ],
        [
            -0.0,
            -0.0,
            -0.0,
            1.0,
        ],
        [
            -0.0,
            -0.0,
            -0.0,
            1.0,
        ],
        [
            1.0,
            -0.0,
            -0.0,
            -0.0,
        ],
        [
            0.20257178,
            -0.0,
            0.61394852,
            0.1834797,
        ],
        [
            -0.0,
            1.0,
            -0.0,
            -0.0,
        ],
        [
            -0.0,
            1.0,
            -0.0,
            -0.0,
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
#     arsenal_u = Arsenal(num_kernels=200, n_estimators=5, random_state=0)
#
#     arsenal_u.fit(X_train, y_train)
#     probas = arsenal_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     arsenal_m = Arsenal(num_kernels=200, n_estimators=5, random_state=0)
#
#     arsenal_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = arsenal_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
