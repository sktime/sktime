# -*- coding: utf-8 -*-
"""HIVE-COTE v2 test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.hybrid import HIVECOTEV2
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.datasets import load_basic_motions, load_unit_test


def test_hivecote_v2_on_unit_test_data():
    """Test of HIVECOTEV2 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train HIVE-COTE v2
    hc2 = HIVECOTEV2(
        stc_params={
            "estimator": RandomForestClassifier(n_estimators=3),
            "n_shapelet_samples": 50,
            "max_shapelets": 5,
            "batch_size": 10,
        },
        drcif_params={"n_estimators": 3, "n_intervals": 2, "att_subsample_size": 2},
        arsenal_params={"num_kernels": 50, "n_estimators": 3},
        tde_params={
            "n_parameter_samples": 5,
            "max_ensemble_size": 3,
            "randomly_selected_params": 3,
        },
        random_state=0,
    )
    hc2.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = hc2.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, hivecote_v2_unit_test_probas, decimal=2)


def test_contracted_hivecote_v2_on_unit_test_data():
    """Test of contracted HIVECOTEV2 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted HIVE-COTE v2
    hc2 = HIVECOTEV2(
        stc_params={
            "estimator": RotationForest(contract_max_n_estimators=3),
            "contract_max_n_shapelet_samples": 50,
            "max_shapelets": 5,
            "batch_size": 10,
        },
        drcif_params={
            "contract_max_n_estimators": 3,
            "n_intervals": 2,
            "att_subsample_size": 2,
        },
        arsenal_params={"contract_max_n_estimators": 3},
        tde_params={
            "contract_max_n_parameter_samples": 5,
            "max_ensemble_size": 3,
            "randomly_selected_params": 3,
        },
        time_limit_in_minutes=2,
        random_state=0,
    )
    hc2.fit(X_train, y_train)


def test_hivecote_v2_on_basic_motions():
    """Test of HIVEVOTEV2 on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 15, replace=False)

    # train HIVE-COTE v2
    hc2 = HIVECOTEV2(
        stc_params={
            "estimator": RandomForestClassifier(n_estimators=3),
            "n_shapelet_samples": 50,
            "max_shapelets": 5,
            "batch_size": 10,
        },
        drcif_params={"n_estimators": 3, "n_intervals": 2, "att_subsample_size": 2},
        arsenal_params={"num_kernels": 50, "n_estimators": 3},
        tde_params={
            "n_parameter_samples": 5,
            "max_ensemble_size": 3,
            "randomly_selected_params": 3,
        },
        random_state=0,
    )
    hc2.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = hc2.predict_proba(X_test.iloc[indices[:10]])
    testing.assert_array_almost_equal(
        probas, hivecote_v2_basic_motions_probas, decimal=2
    )


hivecote_v2_unit_test_probas = np.array(
    [
        [0.0, 1.0],
        [0.4563469353217273, 0.5436530646782726],
        [0.03794465153858766, 0.9620553484614124],
        [1.0, 0.0],
        [0.7189938202882327, 0.28100617971176733],
        [1.0, 0.0],
        [0.8477158354163116, 0.15228416458368835],
        [0.03794465153858766, 0.9620553484614124],
        [0.690228816122276, 0.3097711838777239],
        [1.0, 0.0],
    ]
)
hivecote_v2_basic_motions_probas = np.array(
    [
        [0.0, 0.02216748768472906, 0.02216748768472906, 0.9556650246305418],
        [0.8064844183396478, 0.07006020799124248, 0.0, 0.12345537366910975],
        [0.022167487684729065, 0.0, 0.85802274328765, 0.119809769027621],
        [
            0.07006020799124248,
            0.2803286916694034,
            0.3774291168436061,
            0.27218198349574807,
        ],
        [0.02216748768472906, 0.0, 0.07006020799124248, 0.9077723043240284],
        [0.02216748768472906, 0.0, 0.11439518336070059, 0.8634373289545704],
        [0.7843169306549187, 0.18445539135194305, 0.0, 0.03122767799313823],
        [0.022167487684729065, 0.0, 0.8482957206275515, 0.12953679168771953],
        [
            0.09222769567597153,
            0.7843169306549187,
            0.09222769567597153,
            0.03122767799313823,
        ],
        [0.0, 0.9466048343221327, 0.02216748768472906, 0.03122767799313823],
    ]
)
