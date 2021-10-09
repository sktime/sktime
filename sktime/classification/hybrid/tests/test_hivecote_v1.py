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
            "estimator": RotationForest(n_estimators=5),
            "n_shapelet_samples": 250,
            "max_shapelets": 20,
            "batch_size": 50,
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
            0.3522528187125138,
            0.6477471812874862,
        ],
        [
            0.6300459005582643,
            0.3699540994417357,
        ],
        [
            0.18158252728883603,
            0.8184174727111639,
        ],
        [
            0.7432415438624587,
            0.25675845613754134,
        ],
        [
            0.7399242195263688,
            0.2600757804736313,
        ],
        [
            0.9726484505347228,
            0.027351549465277212,
        ],
        [
            0.736757747326386,
            0.26324225267361395,
        ],
        [
            0.13589070320833677,
            0.8641092967916633,
        ],
        [
            0.7249012692472365,
            0.2750987307527635,
        ],
        [
            0.861186895466627,
            0.13881310453337295,
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
#             "estimator": RotationForest(n_estimators=5),
#             "n_shapelet_samples": 250,
#             "max_shapelets": 20,
#             "batch_size": 50,
#         },
#         tsf_params={"n_estimators": 10},
#         rise_params={"n_estimators": 10},
#         cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
#     )
#
#     hc1.fit(X_train.iloc[indices], y_train[indices])
#     probas = hc1.predict_proba(X_test.iloc[indices])
#     print_array(probas)
