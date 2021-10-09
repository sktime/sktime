# -*- coding: utf-8 -*-
"""HIVE-COTE v2 test code."""
import numpy as np
from numpy import testing

from sktime.classification.hybrid import HIVECOTEV2
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.datasets import load_basic_motions, load_unit_test


def test_hivecote_v2_on_unit_test():
    """Test of HIVECOTEV2 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train HIVE-COTE v2
    hc2 = HIVECOTEV2(
        random_state=0,
        stc_params={
            "estimator": RotationForest(n_estimators=5),
            "n_shapelet_samples": 100,
            "max_shapelets": 10,
            "batch_size": 30,
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
    testing.assert_array_equal(probas, hivecote_v2_unit_test_probas)


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
#             "estimator": RotationForest(contract_max_n_estimators=10),
#             "contract_max_n_shapelet_samples": 100,
#             "max_shapelets": 10,
#             "batch_size": 30,
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
            "estimator": RotationForest(n_estimators=5),
            "n_shapelet_samples": 100,
            "max_shapelets": 10,
            "batch_size": 30,
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
    testing.assert_array_equal(probas, stc_basic_motions_probas)


hivecote_v2_unit_test_probas = np.array(
    [
        [
            0.14367820338807205,
            0.8563217966119279,
        ],
        [
            0.8955021437498036,
            0.10449785625019643,
        ],
        [
            0.03918034713787564,
            0.9608196528621243,
        ],
        [
            0.9133207479454599,
            0.08667925205453997,
        ],
        [
            0.8114216412721086,
            0.18857835872789155,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.7722412941342328,
            0.22775870586576716,
        ],
        [
            0.07836069427575128,
            0.9216393057242487,
        ],
        [
            0.7685129514621234,
            0.23148704853787663,
        ],
        [
            0.8897823355478597,
            0.11021766445214023,
        ],
    ]
)
stc_basic_motions_probas = np.array(
    [
        [
            0.008279263436621997,
            0.0,
            0.04042609099913083,
            0.9512946455642473,
        ],
        [
            0.9096744918362913,
            0.03311705374648798,
            0.0,
            0.05720845441722073,
        ],
        [
            0.16622810510460256,
            0.04042609099913083,
            0.727858086042424,
            0.06548771785384273,
        ],
        [
            0.16170436399652333,
            0.7978695450043458,
            0.04042609099913083,
            0.0,
        ],
        [
            0.0,
            0.09785838196119748,
            0.04042609099913083,
            0.8617155270396717,
        ],
        [
            0.04892919098059874,
            0.04892919098059874,
            0.056984617872374826,
            0.8451570001664277,
        ],
        [
            0.926456855254381,
            0.008279263436621996,
            0.0,
            0.06526388130899681,
        ],
        [
            0.12366963195601641,
            0.16419956120112308,
            0.7038515434062385,
            0.008279263436621996,
        ],
        [
            0.18654215430638932,
            0.7968993188203666,
            0.008279263436621996,
            0.008279263436621996,
        ],
        [
            0.05698461787237482,
            0.9264568552543812,
            0.0,
            0.01655852687324399,
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
#             "estimator": RotationForest(n_estimators=5),
#             "n_shapelet_samples": 100,
#             "max_shapelets": 10,
#             "batch_size": 30,
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
#             "estimator": RotationForest(n_estimators=5),
#             "n_shapelet_samples": 100,
#             "max_shapelets": 10,
#             "batch_size": 30,
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
