# -*- coding: utf-8 -*-
"""WEASEL test code."""
import numpy as np
from numpy import testing

from sktime.classification.dictionary_based import WEASEL
from sktime.datasets import load_unit_test


def test_weasel_on_unit_test_data():
    """Test of WEASEL on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train WEASEL
    weasel = WEASEL(random_state=0, window_inc=4)
    weasel.fit(X_train, y_train)

    # assert probabilities are the same
    probas = weasel.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, weasel_unit_test_probas, decimal=2)


weasel_unit_test_probas = np.array(
    [
        [
            0.23770726036729473,
            0.7622927396327053,
        ],
        [
            0.6074186756627806,
            0.3925813243372193,
        ],
        [
            0.08879116928712083,
            0.9112088307128792,
        ],
        [
            0.9386121802737345,
            0.06138781972626546,
        ],
        [
            0.9260477798507328,
            0.07395222014926714,
        ],
        [
            0.9219499870726096,
            0.07805001292739039,
        ],
        [
            0.21186168283043794,
            0.7881383171695621,
        ],
        [
            0.12450550945199701,
            0.875494490548003,
        ],
        [
            0.8866777696427675,
            0.11332223035723252,
        ],
        [
            0.9240701105998846,
            0.0759298894001154,
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
#     weasel = WEASEL(random_state=0, window_inc=4)
#
#     weasel.fit(X_train, y_train)
#     probas = weasel.predict_proba(X_test.iloc[indices])
#     print_array(probas)
