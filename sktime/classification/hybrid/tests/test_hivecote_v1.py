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
            "n_shapelet_samples": 100,
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
            0.26806096567079934,
            0.7319390343292006,
        ],
        [
            0.7123998602231357,
            0.28760013977686427,
        ],
        [
            0.1400531000605982,
            0.8599468999394018,
        ],
        [
            0.7373484114834657,
            0.2626515885165343,
        ],
        [
            0.6733038279724523,
            0.32669617202754764,
        ],
        [
            0.969808586438017,
            0.03019141356198297,
        ],
        [
            0.7301914135619829,
            0.26980858643801703,
        ],
        [
            0.08961717287603407,
            0.910382827123966,
        ],
        [
            0.6171038978608845,
            0.38289610213911557,
        ],
        [
            0.8071569979214828,
            0.19284300207851726,
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
#             "n_shapelet_samples": 100,
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
