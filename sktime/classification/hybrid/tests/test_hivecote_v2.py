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
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

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
            0.0,
            0.0,
            0.03732718894009217,
            0.9626728110599079,
        ],
        [
            0.9533410138248848,
            0.046658986175115214,
            0.0,
            0.0,
        ],
        [
            0.05713993544930875,
            0.0,
            0.886869281140553,
            0.055990783410138245,
        ],
        [
            0.05599078341013825,
            0.5592140336841426,
            0.3567997912006501,
            0.027995391705069126,
        ],
        [
            0.0,
            0.0,
            0.055990783410138245,
            0.9440092165898617,
        ],
        [
            0.0,
            0.00933179723502304,
            0.06532258064516129,
            0.9253456221198156,
        ],
        [
            0.9720046082949308,
            0.027995391705069123,
            0.0,
            0.0,
        ],
        [
            0.016051824124423963,
            0.03128480637096774,
            0.8178584400557337,
            0.13480492944887468,
        ],
        [
            0.06532258064516129,
            0.8377572994566552,
            0.08758832266316038,
            0.00933179723502304,
        ],
        [
            0.06532258064516129,
            0.8470890966916783,
            0.08758832266316038,
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
#         verbose=1
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
#     probas = hc2_m.predict_proba(X_test.iloc[indices])
#     print_array(probas)
