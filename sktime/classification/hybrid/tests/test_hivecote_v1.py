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
            0.2896146167253226,
            0.7103853832746773,
        ],
        [
            0.6447393690503758,
            0.3552606309496243,
        ],
        [
            0.15131417882399237,
            0.8486858211760075,
        ],
        [
            0.7590323281846022,
            0.24096767181539777,
        ],
        [
            0.6898381899763045,
            0.31016181002369553,
        ],
        [
            0.9673810222787618,
            0.032618977721238215,
        ],
        [
            0.7512998662621914,
            0.2487001337378087,
        ],
        [
            0.0968229115699514,
            0.9031770884300486,
        ],
        [
            0.671922083209323,
            0.328077916790677,
        ],
        [
            0.8344539284716492,
            0.16554607152835074,
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
