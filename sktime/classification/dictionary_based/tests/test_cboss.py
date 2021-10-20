# -*- coding: utf-8 -*-
"""cBOSS test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.dictionary_based import ContractableBOSS
from sktime.datasets import load_unit_test


def test_cboss_on_unit_test():
    """Test of cBOSS on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train cBOSS
    cboss = ContractableBOSS(
        n_parameter_samples=25, max_ensemble_size=5, random_state=0
    )
    cboss.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = cboss.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, cboss_unit_test_probas)

    # test train estimate
    train_probas = cboss._get_train_probs(X_train, y_train)
    train_preds = cboss.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


cboss_unit_test_probas = np.array(
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
