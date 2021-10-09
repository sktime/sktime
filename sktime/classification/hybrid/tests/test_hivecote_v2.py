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
            "max_shapelets": 20,
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
#             "max_shapelets": 20,
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
            "max_shapelets": 20,
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
            0.26472960379372157,
            0.7352703962062784,
        ],
        [
            0.9079104072552392,
            0.09208959274476083,
        ],
        [
            0.1726400110489607,
            0.8273599889510392,
        ],
        [
            0.9689208428220419,
            0.03107915717795815,
        ],
        [
            0.7410094106134945,
            0.25899058938650565,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.7755374128232866,
            0.2244625871767135,
        ],
        [
            0.13811200883916858,
            0.8618879911608315,
        ],
        [
            0.8175594247692761,
            0.18244057523072388,
        ],
        [
            0.9481774238722471,
            0.051822576127752755,
        ],
    ]
)
stc_basic_motions_probas = np.array(
    [
        [
            0.008279263436621996,
            0.0,
            0.0,
            0.991720736563378,
        ],
        [
            0.8692484008371605,
            0.07354314474561881,
            0.0,
            0.05720845441722073,
        ],
        [
            0.16622810510460254,
            0.0,
            0.7278580860424239,
            0.10591380885297355,
        ],
        [
            0.1212782729973925,
            0.8382956360034767,
            0.0,
            0.04042609099913083,
        ],
        [
            0.04042609099913083,
            0.09785838196119748,
            0.0,
            0.8617155270396717,
        ],
        [
            0.04892919098059874,
            0.04892919098059874,
            0.056984617872374826,
            0.8451570001664277,
        ],
        [
            0.9668829462535119,
            0.008279263436621996,
            0.0,
            0.024837790309865986,
        ],
        [
            0.08324354095688558,
            0.20462565220025392,
            0.7038515434062385,
            0.008279263436621996,
        ],
        [
            0.1461160633072585,
            0.7968993188203667,
            0.008279263436621997,
            0.04870535443575283,
        ],
        [
            0.05698461787237482,
            0.8860307642552503,
            0.04042609099913083,
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
#             "max_shapelets": 20,
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
#             "max_shapelets": 20,
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
