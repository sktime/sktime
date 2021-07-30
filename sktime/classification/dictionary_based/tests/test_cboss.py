# -*- coding: utf-8 -*-
import numpy as np
from numpy import testing

from sktime.classification.dictionary_based import ContractableBOSS
from sktime.datasets import load_gunpoint, load_italy_power_demand


def test_cboss_on_gunpoint():
    # load gunpoint data
    X_train, y_train = load_gunpoint(split="train", return_X_y=True)
    X_test, y_test = load_gunpoint(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(10)

    # train cBOSS
    cboss = ContractableBOSS(
        n_parameter_samples=50, max_ensemble_size=10, random_state=0
    )
    cboss.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = cboss.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cboss_gunpoint_probas)


def test_cboss_on_power_demand():
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train cBOSS
    cboss = ContractableBOSS(
        n_parameter_samples=50, max_ensemble_size=10, random_state=0
    )
    cboss.fit(X_train, y_train)

    score = cboss.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.9


cboss_gunpoint_probas = np.array(
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
            0.6608256880733945,
            0.33917431192660547,
        ],
        [
            0.4999999999999999,
            0.4999999999999999,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.2797247706422018,
            0.7202752293577981,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.3898623853211009,
            0.6101376146788989,
        ],
        [
            0.2797247706422018,
            0.7202752293577981,
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
#     cboss = ContractableBOSS(
#         n_parameter_samples=50, max_ensemble_size=10, random_state=0
#     )
#
#     cboss.fit(X_train.iloc[indices], y_train[indices])
#     probas = cboss.predict_proba(X_test.iloc[indices])
#     print_array(probas)
