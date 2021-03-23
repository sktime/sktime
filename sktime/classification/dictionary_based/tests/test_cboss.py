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
    assert score >= 0.88


cboss_gunpoint_probas = np.array(
    [
        [
            0.07456846950517836,
            0.9254315304948215,
        ],
        [
            0.21271576524741084,
            0.7872842347525892,
        ],
        [
            0.5000000000000001,
            0.5000000000000001,
        ],
        [
            0.2982738780207134,
            0.7017261219792866,
        ],
        [
            0.0,
            0.9999999999999999,
        ],
        [
            0.07456846950517836,
            0.9254315304948215,
        ],
        [
            0.07456846950517836,
            0.9254315304948215,
        ],
        [
            0.22370540851553505,
            0.7762945914844649,
        ],
        [
            0.07456846950517836,
            0.9254315304948215,
        ],
        [
            0.13814729574223247,
            0.8618527042577675,
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
