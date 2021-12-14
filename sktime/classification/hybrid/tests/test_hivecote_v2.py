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
            0.15029100580102842,
            0.8497089941989715,
        ],
        [
            0.8979197370897057,
            0.10208026291029436,
        ],
        [
            0.02410537144536704,
            0.9758946285546329,
        ],
        [
            0.8622633828925127,
            0.13773661710748725,
        ],
        [
            0.7221219503428462,
            0.2778780496571538,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.7703326932335803,
            0.2296673067664197,
        ],
        [
            0.24300585396918709,
            0.7569941460308129,
        ],
        [
            0.7292763937385132,
            0.2707236062614868,
        ],
        [
            0.7703326932335803,
            0.2296673067664197,
        ],
    ]
)
hivecote_v2_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.8521477383024575,
            0.02493038764697154,
            0.04986077529394308,
            0.07306109875662808,
        ],
        [
            0.07178405432539303,
            0.0,
            0.8052940716240359,
            0.12292187405057114,
        ],
        [
            0.11012950661790163,
            0.8898704933820984,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.14612219751325614,
            0.0,
            0.8538778024867439,
        ],
        [
            0.11566065824209311,
            0.07306109875662807,
            0.024930387646971537,
            0.7863478553543073,
        ],
        [
            0.9324700528675633,
            0.0,
            0.0,
            0.06752994713243658,
        ],
        [
            0.035947609742638574,
            0.11568934643598923,
            0.7985022686176774,
            0.049860775203694634,
        ],
        [
            0.11739072242637963,
            0.7974101586026903,
            0.04259955948546504,
            0.04259955948546504,
        ],
        [
            0.024930387646971537,
            0.9324700528675633,
            0.04259955948546504,
            0.0,
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
