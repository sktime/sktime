# -*- coding: utf-8 -*-
"""HIVE-COTE v1 test code."""
import numpy as np
from numpy import testing

from sktime.classification.hybrid import HIVECOTEV1
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.datasets import load_unit_test


def test_hivecote_v1_on_unit_test():
    """Test of HIVECOTEV1 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train HIVE-COTE v1
    hc1 = HIVECOTEV1(
        random_state=0,
        stc_params={
            "estimator": RotationForest(n_estimators=10),
            "n_shapelets_considered": 100,
            "max_shapelets": 10,
            "batch_size": 30,
        },
        tsf_params={"n_estimators": 10},
        rise_params={"n_estimators": 10},
        cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
    )
    hc1.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = hc1.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, hivecote_v1_unit_test_probas)


hivecote_v1_unit_test_probas = np.array(
    [
        [
            0.323377079547267,
            0.676622920452733,
        ],
        [
            0.6487384919521377,
            0.35126150804786227,
        ],
        [
            0.1711756032669395,
            0.8288243967330604,
        ],
        [
            0.7348118636305183,
            0.26518813636948163,
        ],
        [
            0.630748802459115,
            0.3692511975408849,
        ],
        [
            0.9641023800122052,
            0.03589761998779481,
        ],
        [
            0.7169975230642208,
            0.2830024769357792,
        ],
        [
            0.11120723696018955,
            0.8887927630398105,
        ],
        [
            0.6296412142351372,
            0.37035878576486286,
        ],
        [
            0.8178143405662975,
            0.1821856594337024,
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
#     hc1 = HIVECOTEV1(
#         random_state=0,
#         stc_params={
#             "estimator": RotationForest(n_estimators=10),
#             "n_shapelets_considered": 100,
#             "max_shapelets": 10,
#             "batch_size": 30,
#         },
#         tsf_params={"n_estimators": 10},
#         rise_params={"n_estimators": 10},
#         cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
#     )
#
#     hc1.fit(X_train.iloc[indices], y_train[indices])
#     probas = hc1.predict_proba(X_test.iloc[indices])
#     print_array(probas)
