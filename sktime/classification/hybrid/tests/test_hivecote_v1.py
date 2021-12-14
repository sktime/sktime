# -*- coding: utf-8 -*-
"""HIVE-COTE v1 test code."""
import numpy as np
from numpy import testing

from sktime.classification.hybrid import HIVECOTEV1
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.datasets import load_unit_test


def test_hivecote_v1_on_unit_test_data():
    """Test of HIVECOTEV1 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train HIVE-COTE v1
    hc1 = HIVECOTEV1(
        random_state=0,
        stc_params={
            "estimator": RotationForest(n_estimators=3),
            "n_shapelet_samples": 500,
            "max_shapelets": 20,
            "batch_size": 100,
        },
        tsf_params={"n_estimators": 10},
        rise_params={"n_estimators": 10},
        cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
    )
    hc1.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = hc1.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, hivecote_v1_unit_test_probas, decimal=2)


hivecote_v1_unit_test_probas = np.array(
    [
        [
            0.3007483371934601,
            0.6992516628065399,
        ],
        [
            0.6805717631680364,
            0.3194282368319636,
        ],
        [
            0.14613851420128854,
            0.8538614857987114,
        ],
        [
            0.7429919663642466,
            0.2570080336357534,
        ],
        [
            0.7211218145853933,
            0.2788781854146068,
        ],
        [
            0.9706710464570624,
            0.02932895354293759,
        ],
        [
            0.7360394181135617,
            0.26396058188643834,
        ],
        [
            0.11731581417175037,
            0.8826841858282497,
        ],
        [
            0.6848403127917707,
            0.31515968720822923,
        ],
        [
            0.8309788269930594,
            0.16902117300694067,
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
#             "estimator": RotationForest(n_estimators=3),
#             "n_shapelet_samples": 500,
#             "max_shapelets": 20,
#             "batch_size": 100,
#         },
#         tsf_params={"n_estimators": 10},
#         rise_params={"n_estimators": 10},
#         cboss_params={"n_parameter_samples": 25, "max_ensemble_size": 5},
#     )
#
#     hc1.fit(X_train.iloc[indices], y_train[indices])
#     probas = hc1.predict_proba(X_test.iloc[indices])
#     print_array(probas)
