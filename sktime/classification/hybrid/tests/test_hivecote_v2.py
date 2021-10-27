# -*- coding: utf-8 -*-
"""HIVE-COTE v2 test code."""
import numpy as np
from numpy import testing

from sktime.classification.hybrid import HIVECOTEV2
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.datasets import load_basic_motions, load_unit_test


def test_hivecote_v2_on_unit_test_data():
    """Test of HIVECOTEV2 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train HIVE-COTE v2
    hc2 = HIVECOTEV2(
        random_state=0,
        stc_params={
            "estimator": RotationForest(n_estimators=3),
            "n_shapelet_samples": 500,
            "max_shapelets": 20,
            "batch_size": 100,
        },
        drcif_params={"n_estimators": 10},
        arsenal_params={"num_kernels": 100, "n_estimators": 5},
        tde_params={
            "n_parameter_samples": 10,
            "max_ensemble_size": 5,
            "randomly_selected_params": 5,
        },
    )
    hc2.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = hc2.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, hivecote_v2_unit_test_probas, decimal=2)


# def test_contracted_hivecote_v2_on_unit_test_data():
#     """Test of contracted HIVECOTEV2 on unit test data."""
#     # load unit test data
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#
#     # train contracted HIVE-COTE v2
#     hc2 = HIVECOTEV2(
#         time_limit_in_minutes=2,
#         random_state=0,
#         stc_params={
#             "estimator": RotationForest(contract_max_n_estimators=3),
#             "contract_max_n_shapelet_samples": 500,
#             "max_shapelets": 20,
#             "batch_size": 100,
#         },
#         drcif_params={"contract_max_n_estimators": 10},
#         arsenal_params={"contract_max_n_estimators": 5},
#         tde_params={
#             "contract_max_n_parameter_samples": 10,
#             "max_ensemble_size": 5,
#             "randomly_selected_params": 5,
#         },
#     )
#     hc2.fit(X_train, y_train)
#
#     assert accuracy_score(y_test, hc2.predict(X_test)) >= 0.75


def test_hivecote_v2_on_basic_motions():
    """Test of HIVEVOTEV2 on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 15, replace=False)

    # train HIVE-COTE v2
    hc2 = HIVECOTEV2(
        random_state=0,
        stc_params={
            "estimator": RotationForest(n_estimators=3),
            "n_shapelet_samples": 500,
            "max_shapelets": 20,
            "batch_size": 100,
        },
        drcif_params={"n_estimators": 10},
        arsenal_params={"num_kernels": 100, "n_estimators": 5},
        tde_params={
            "n_parameter_samples": 25,
            "max_ensemble_size": 5,
            "randomly_selected_params": 10,
        },
    )
    hc2.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = hc2.predict_proba(X_test.iloc[indices[:10]])
    testing.assert_array_almost_equal(
        probas, hivecote_v2_basic_motions_probas, decimal=2
    )


hivecote_v2_unit_test_probas = np.array(
    [
        [
            0.1590380252005275,
            0.8409619747994725,
        ],
        [
            0.8251178715236753,
            0.1748821284763248,
        ],
        [
            0.04336889582791222,
            0.9566311041720877,
        ],
        [
            0.9609630258397076,
            0.039036974160292476,
        ],
        [
            0.8481704636216507,
            0.1518295363783492,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.8048015677937387,
            0.19519843220626143,
        ],
        [
            0.08673779165582444,
            0.9132622083441755,
        ],
        [
            0.8575833142582955,
            0.1424166857417046,
        ],
        [
            0.9349082552774752,
            0.06509174472252473,
        ],
    ]
)
hivecote_v2_basic_motions_probas = np.array(
    [
        [
            0.008897771212582005,
            0.0,
            0.0,
            0.9911022287874179,
        ],
        [
            0.8554183024420393,
            0.03559108485032802,
            0.0,
            0.10899061270763263,
        ],
        [
            0.17864628413433553,
            0.0,
            0.7509736953290359,
            0.0703800205366286,
        ],
        [
            0.09501672676717207,
            0.9049832732328279,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.10516895622292918,
            0.0,
            0.8948310437770708,
        ],
        [
            0.05258447811146459,
            0.05258447811146459,
            0.01779554242516401,
            0.8770355013519068,
        ],
        [
            0.9644089151496719,
            0.008897771212582005,
            0.0,
            0.026693313637746012,
        ],
        [
            0.0894623039874875,
            0.17646619653523835,
            0.7251737282646921,
            0.008897771212582005,
        ],
        [
            0.07420167702133204,
            0.8604944171699178,
            0.008897771212582005,
            0.056406134596168035,
        ],
        [
            0.01779554242516401,
            0.964408915149672,
            0.0,
            0.01779554242516401,
        ],
    ]
)


# def print_array(array):
#     print("[")
#     for sub_array in array:
#         print("[")
#         for value in sub_array:
#             print(value.astype(str), end="")
#             print(", ")
#         print("],")
#     print("]")
#
# if __name__ == "__main__":
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     X_test, y_test = load_unit_test(split="test", return_X_y=True)
#     indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
#
#     hc2_u = HIVECOTEV2(
#         random_state=0,
#         stc_params={
#             "estimator": RotationForest(n_estimators=3),
#             "n_shapelet_samples": 500,
#             "max_shapelets": 20,
#             "batch_size": 100,
#         },
#         drcif_params={"n_estimators": 10},
#         arsenal_params={"num_kernels": 100, "n_estimators": 5},
#         tde_params={
#             "n_parameter_samples": 10,
#             "max_ensemble_size": 5,
#             "randomly_selected_params": 5,
#         },
#     )
#
#     hc2_u.fit(X_train.iloc[indices], y_train[indices])
#     probas = hc2_u.predict_proba(X_test.iloc[indices])
#     print_array(probas)
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     X_test, y_test = load_basic_motions(split="test", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 15, replace=False)
#
#     hc2_m = HIVECOTEV2(
#         random_state=0,
#         stc_params={
#             "estimator": RotationForest(n_estimators=3),
#             "n_shapelet_samples": 500,
#             "max_shapelets": 20,
#             "batch_size": 100,
#         },
#         drcif_params={"n_estimators": 10},
#         arsenal_params={"num_kernels": 100, "n_estimators": 5},
#         tde_params={
#             "n_parameter_samples": 25,
#             "max_ensemble_size": 5,
#             "randomly_selected_params": 10,
#         },
#     )
#
#     hc2_m.fit(X_train.iloc[indices], y_train[indices])
#     probas = hc2_m.predict_proba(X_test.iloc[indices[:10]])
#     print_array(probas)
