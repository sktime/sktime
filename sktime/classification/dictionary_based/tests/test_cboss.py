# -*- coding: utf-8 -*-
"""cBOSS test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.dictionary_based import ContractableBOSS
from sktime.datasets import load_unit_test


def test_cboss_on_unit_test_data():
    """Test of cBOSS on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train cBOSS
    cboss = ContractableBOSS(
        n_parameter_samples=25,
        max_ensemble_size=5,
        random_state=0,
        save_train_predictions=True,
    )
    cboss.fit(X_train, y_train)

    # assert probabilities are the same
    probas = cboss.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, cboss_unit_test_probas, decimal=2)

    # test train estimate
    train_probas = cboss._get_train_probs(X_train, y_train)
    train_preds = cboss.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


cboss_unit_test_probas = np.array(
    [
        [
            0.12929747869474983,
            0.8707025213052502,
        ],
        [
            0.5646487393473748,
            0.43535126065262497,
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
            0.5044553362476266,
            0.4955446637523732,
        ],
        [
            0.8707025213052502,
            0.12929747869474983,
        ],
        [
            0.7477723318761866,
            0.2522276681238133,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.5646487393473748,
            0.43535126065262497,
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
#     cboss = ContractableBOSS(
#         n_parameter_samples=25, max_ensemble_size=5, random_state=0
#     )
#
#     cboss.fit(X_train, y_train)
#     probas = cboss.predict_proba(X_test.iloc[indices])
#     print_array(probas)
