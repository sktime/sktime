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


def test_contracted_hivecote_v2_on_unit_test_data():
    """Test of contracted HIVECOTEV2 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted HIVE-COTE v2
    hc2 = HIVECOTEV2(
        time_limit_in_minutes=2,
        random_state=0,
        stc_params={
            "estimator": RotationForest(contract_max_n_estimators=3),
            "contract_max_n_shapelet_samples": 500,
            "max_shapelets": 20,
            "batch_size": 100,
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


def test_hivecote_v2_on_basic_motions():
    """Test of HIVEVOTEV2 on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
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
            0.12350161813575242,
            0.8764983818642476,
        ],
        [
            0.8154775095336717,
            0.18452249046632827,
        ],
        [
            0.01980858643801703,
            0.980191413561983,
        ],
        [
            0.8868149494465434,
            0.11318505055345655,
        ],
        [
            0.7716537420575929,
            0.2283462579424072,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.710632869727017,
            0.28936713027298305,
        ],
        [
            0.15846869150413623,
            0.8415313084958638,
        ],
        [
            0.8187542484785999,
            0.18124575152140013,
        ],
        [
            0.710632869727017,
            0.28936713027298305,
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
            0.8596164476714715,
            0.023671037146314373,
            0.047342074292628746,
            0.06937044088958527,
        ],
        [
            0.06815790594639394,
            0.0,
            0.8151295788713919,
            0.11671251518221402,
        ],
        [
            0.19552868438064613,
            0.8044713156193538,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.13874088177917057,
            0.0,
            0.8612591182208295,
        ],
        [
            0.06937044088958529,
            0.06937044088958529,
            0.023671037146314376,
            0.8375880810745151,
        ],
        [
            0.9763289628536855,
            0.0,
            0.0,
            0.023671037146314376,
        ],
        [
            0.034131727820139936,
            0.10984533638016583,
            0.8086808615884263,
            0.04734207421126777,
        ],
        [
            0.18558487626183096,
            0.7571292413267251,
            0.05728588241144392,
            0.0,
        ],
        [
            0.08095691955775829,
            0.9190430804422416,
            0.0,
            0.0,
        ],
    ]
)
