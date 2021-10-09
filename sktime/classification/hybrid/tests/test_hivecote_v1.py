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
            "n_shapelet_samples": 200,
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
            0.24284662085140493,
            0.757153379148595,
        ],
        [
            0.7394520984193731,
            0.26054790158062685,
        ],
        [
            0.12687942835828162,
            0.8731205716417183,
        ],
        [
            0.797944642793013,
            0.20205535720698695,
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
            0.7914608462569405,
            0.20853915374305954,
        ],
        [
            0.08118760427778235,
            0.9188123957222177,
        ],
        [
            0.6154950713861278,
            0.3845049286138723,
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
#             "n_shapelet_samples": 200,
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
