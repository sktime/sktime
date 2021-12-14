# -*- coding: utf-8 -*-
"""HIVE-COTE v2 test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.hybrid import HIVECOTEV2
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
            "estimator": RandomForestClassifier(n_estimators=3),
            "n_shapelet_samples": 500,
            "max_shapelets": 20,
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
            "estimator": RandomForestClassifier(n_estimators=3),
            "n_shapelet_samples": 500,
            "max_shapelets": 20,
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
            0.13781021446256755,
            0.8621897855374324,
        ],
        [
            0.8327183709230067,
            0.1672816290769933,
        ],
        [
            0.022103560960819325,
            0.9778964390391807,
        ],
        [
            0.8737016054835566,
            0.12629839451644334,
        ],
        [
            0.7451981014193845,
            0.2548018985806156,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.7894052233410231,
            0.21059477665897688,
        ],
        [
            0.1768284876865546,
            0.8231715123134454,
        ],
        [
            0.797755557662911,
            0.20224444233708905,
        ],
        [
            0.6420481502688942,
            0.3579518497311057,
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
            0.7940682270013237,
            0.0963273999932546,
            0.04445879999688674,
            0.06514557300853482,
        ],
        [
            0.13810488507439375,
            0.0,
            0.7522907419201846,
            0.10960437300542158,
        ],
        [
            0.17042539998806586,
            0.7554766000171228,
            0.07409799999481124,
            0.0,
        ],
        [
            0.0,
            0.13029114601706968,
            0.0,
            0.8697088539829304,
        ],
        [
            0.06514557300853484,
            0.06514557300853484,
            0.09632739999325461,
            0.7733814539896757,
        ],
        [
            0.8295746000119341,
            0.0,
            0.07409799999481124,
            0.09632739999325461,
        ],
        [
            0.03205300324364694,
            0.10315542599678769,
            0.8203327708344312,
            0.044458799925134254,
        ],
        [
            0.21488419998495262,
            0.7110178000202361,
            0.0,
            0.07409799999481124,
        ],
        [
            0.09632739999325461,
            0.9036726000067453,
            0.0,
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
#             "estimator": RandomForestClassifier(n_estimators=3),
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
#             "estimator": RandomForestClassifier(n_estimators=3),
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
