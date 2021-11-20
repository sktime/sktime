# -*- coding: utf-8 -*-
"""ElasticEnsemble test code."""
import numpy as np
from numpy import testing

from sktime.classification.distance_based import ElasticEnsemble
from sktime.datasets import load_unit_test


def test_ee_on_unit_test_data():
    """Test of ElasticEnsemble on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train EE
    ee = ElasticEnsemble(
        proportion_of_param_options=0.1,
        proportion_train_for_test=0.1,
        random_state=0,
    )
    ee.fit(X_train, y_train)

    # assert probabilities are the same
    probas = ee.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, ee_unit_test_probas, decimal=2)


ee_unit_test_probas = np.array(
    [
        [
            0.08264462809917356,
            0.9173553719008264,
        ],
        [
            0.8429752066115702,
            0.15702479338842976,
        ],
        [
            0.08264462809917356,
            0.9173553719008264,
        ],
        [
            0.8429752066115702,
            0.15702479338842976,
        ],
        [
            0.5619834710743802,
            0.4380165289256198,
        ],
        [
            0.8429752066115702,
            0.15702479338842976,
        ],
        [
            0.7024793388429752,
            0.29752066115702475,
        ],
        [
            0.08264462809917356,
            0.9173553719008264,
        ],
        [
            0.7024793388429752,
            0.29752066115702475,
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
#     ee = ElasticEnsemble(
#         proportion_of_param_options=0.1,
#         proportion_train_for_test=0.1,
#         random_state=0,
#     )
#
#     ee.fit(X_train, y_train)
#     probas = ee.predict_proba(X_test.iloc[indices])
#     print_array(probas)
