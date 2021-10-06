# -*- coding: utf-8 -*-
"""HIVE-COTE v2 test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

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
            "estimator": RotationForest(n_estimators=10),
            "n_shapelet_samples": 100,
            "max_shapelets": 10,
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
    testing.assert_array_equal(probas, hivecote_v2_unit_test_probas)


def test_contracted_hivecote_v2_on_unit_test_data():
    """Test of contracted HIVECOTEV2 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)

    # train contracted HIVE-COTE v2
    hc2 = HIVECOTEV2(
        time_limit_in_minutes=2,
        random_state=0,
        stc_params={
            "estimator": RotationForest(contract_max_n_estimators=10),
            "contract_max_n_shapelet_samples": 100,
            "max_shapelets": 10,
            "batch_size": 30,
        },
        drcif_params={"contract_max_n_estimators": 10},
        arsenal_params={"contract_max_n_estimators": 5},
        tde_params={
            "contract_max_n_parameter_samples": 10,
            "max_ensemble_size": 5,
            "randomly_selected_params": 5,
        },
    )
    hc2.fit(X_train, y_train)

    assert accuracy_score(y_test, hc2.predict(X_test)) >= 0.75


def test_hivecote_v2_on_basic_motions():
    """Test of HIVEVOTEV2 on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    X_test, y_test = load_basic_motions(split="test", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train HIVE-COTE v2
    hc2 = HIVECOTEV2(
        random_state=0,
        stc_params={
            "estimator": RotationForest(n_estimators=10),
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
    probas = hc2.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, stc_basic_motions_probas)


hivecote_v2_unit_test_probas = np.array(
    [
        [
            0.18741725664299355,
            0.8125827433570065,
        ],
        [
            0.8555484852861209,
            0.14445151471387913,
        ],
        [
            0.055760687135946285,
            0.9442393128640536,
        ],
        [
            0.9555674523552165,
            0.04443254764478349,
        ],
        [
            0.7888002048666217,
            0.21119979513337814,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.7650268807477552,
            0.23497311925224482,
        ],
        [
            0.10512390166847667,
            0.8948760983315235,
        ],
        [
            0.8251039618998803,
            0.1748960381001197,
        ],
        [
            0.9259114695521782,
            0.07408853044782188,
        ],
    ]
)
stc_basic_motions_probas = np.array(
    [
        [
            0.062217954872779646,
            0.062217954872779646,
            0.18665386461833894,
            0.6889102256361018,
        ],
        [
            0.7939030244839175,
            0.14387902064330294,
            0.0,
            0.062217954872779646,
        ],
        [
            0.016033348041286605,
            0.0,
            0.8439762534949593,
            0.1399903984637542,
        ],
        [
            0.007777244359097457,
            0.5907814600277655,
            0.3975526734335883,
            0.0038886221795487283,
        ],
        [
            0.062217954872779646,
            0.12443590974555929,
            0.13221315410465675,
            0.6811329812770044,
        ],
        [
            0.0,
            0.003888622179548728,
            0.19831973115698512,
            0.7977916466634661,
        ],
        [
            0.9261161785885742,
            0.07388382141142583,
            0.0,
            0.0,
        ],
        [
            0.06890685710033605,
            0.013036587580412864,
            0.7996644867827061,
            0.11839206853654498,
        ],
        [
            0.011665866538646184,
            0.7612930081269212,
            0.2231525031548839,
            0.003888622179548728,
        ],
        [
            0.07388382141142584,
            0.7651816303064699,
            0.09871659340932465,
            0.06221795487277965,
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
#             "estimator": RotationForest(n_estimators=10),
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
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     hc2_m = HIVECOTEV2(
#         random_state=0,
#         stc_params={
#             "estimator": RotationForest(n_estimators=10),
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
#     probas = hc2_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
