# -*- coding: utf-8 -*-
"""HIVE-COTE test code."""
import numpy as np

from sktime.classification.hybrid import HIVECOTEV1
from sktime.datasets import load_italy_power_demand


# def test_hivecote_v1_on_gunpoint():
#     # load gunpoint data
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
#     # train HIVE-COTE v1
#     hc1 = HIVECOTEV1(
#         random_state=0,
#         stc_params={"n_estimators": 10, "transform_contract_in_mins": 0.1},
#         tsf_params={"n_estimators": 10},
#         rise_params={"n_estimators": 10},
#         cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
#     )
#     hc1.fit(X_train.iloc[indices], y_train[indices])
#
#     # assert probabilities are the same
#     probas = hc1.predict_proba(X_test.iloc[indices])
#     testing.assert_array_equal(probas, hivecote_v1_gunpoint_probas)


def test_hivecote_v1_on_power_demand():
    """Test of HIVE-COTEv1 on italy power demand."""
    # load power demand data
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    indices = np.random.RandomState(0).permutation(100)

    # train HIVE-COTE v1
    hc1 = HIVECOTEV1(
        random_state=0,
        stc_params={"n_estimators": 10, "transform_contract_in_mins": 0.1},
        tsf_params={"n_estimators": 10},
        rise_params={"n_estimators": 10},
        cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
    )
    hc1.fit(X_train, y_train)

    score = hc1.score(X_test.iloc[indices], y_test[indices])
    assert score >= 0.92


# hivecote_v1_gunpoint_probas = np.array(
#     []
# )


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
#     X_train, y_train = load_gunpoint(split="train", return_X_y=True)
#     X_test, y_test = load_gunpoint(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).permutation(10)
#
# hc1 = HIVECOTEV1(
#     random_state=0,
#     stc_params={"n_estimators": 10, "transform_contract_in_mins": 0.1},
#     tsf_params={"n_estimators": 10},
#     rise_params={"n_estimators": 10},
#     cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
# )
#
#     hc1.fit(X_train.iloc[indices], y_train[indices])
#     probas = hc1.predict_proba(X_test.iloc[indices])
#     print_array(probas)
