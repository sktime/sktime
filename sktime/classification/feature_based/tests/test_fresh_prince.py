# -*- coding: utf-8 -*-
"""FreshPRINCE test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.feature_based import FreshPRINCE
from sktime.datasets import load_unit_test


def test_fresh_prince_on_unit_test_data():
    """Test of FreshPRINCE on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train FreshPRINCE classifier
    fp = FreshPRINCE(
        random_state=0,
        default_fc_parameters="minimal",
        n_estimators=10,
        save_transformed_data=True,
    )
    fp.fit(X_train, y_train)

    # assert probabilities are the same
    probas = fp.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, fp_classifier_unit_test_probas, decimal=2)

    # test train estimate
    train_probas = fp._get_train_probs(X_train, y_train)
    train_preds = fp.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


fp_classifier_unit_test_probas = np.array(
    [
        [
            0.2,
            0.8,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.1,
            0.9,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.9,
            0.1,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.8,
            0.2,
        ],
        [
            0.9,
            0.1,
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
# if __name__ == "__main__":
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
#
#     fp = FreshPRINCE(
#         random_state=0,
#         default_fc_parameters="minimal",
#         n_estimators=10,
#     )
#
#     fp.fit(X_train, y_train)
#     probas = fp.predict_proba(X_test.iloc[indices])
#     print_array(probas)
