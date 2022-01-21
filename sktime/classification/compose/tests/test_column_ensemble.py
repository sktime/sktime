# -*- coding: utf-8 -*-
"""Column ensemble test code."""
__author__ = ["TonyBagnall"]


import numpy as np
from numpy import testing

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.feature_based import FreshPRINCE
from sktime.classification.interval_based import DrCIF
from sktime.datasets import load_basic_motions, load_unit_test


def test_col_ens_on_basic_motions():
    """Test of ColumnEnsembleClassifier on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
    fp = FreshPRINCE(
        random_state=0,
        default_fc_parameters="minimal",
        n_estimators=10,
    )
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=10,
        max_ensemble_size=5,
        randomly_selected_params=5,
        random_state=0,
    )
    drcif = DrCIF(n_estimators=10, random_state=0, save_transformed_data=True)
    estimators = [
        ("FreshPrince", fp, [0, 1, 2]),
        ("TDE", tde, [3, 4]),
        ("DrCIF", drcif, [5]),
    ]

    # train column ensemble
    col_ens = ColumnEnsembleClassifier(estimators=estimators)
    col_ens.fit(X_train, y_train)
    # preds = col_ens.predict(X_test.iloc[indices])

    # assert preds[0] == 2
    # assert probabilities are the same
    probas = col_ens.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, col_ens_basic_motions_probas, decimal=2)


def test_col_ens_on_unit_test_data():
    """Test of ColumnEnsembleClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train Column ensemble with a single
    fp = FreshPRINCE(
        random_state=0,
        default_fc_parameters="minimal",
        n_estimators=10,
    )
    estimators = [("FreshPrince", fp, [0])]
    col_ens = ColumnEnsembleClassifier(estimators=estimators)
    col_ens.fit(X_train, y_train)
    # preds = col_ens.predict(X_test.iloc[indices])

    # assert preds[0] == 2
    # assert probabilities are the same
    probas = col_ens.predict_proba(X_test.iloc[indices])

    testing.assert_array_almost_equal(probas, col_ens_unit_test_probas, decimal=2)


col_ens_unit_test_probas = np.array(
    [
        [
            0.2,
            0.8,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.1,
            0.9,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.9,
            0.1,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.8,
            0.2,
        ],
        [
            0.9,
            0.1,
        ],
        [
            1.0,
            0.0,
        ],
    ]
)
col_ens_basic_motions_probas = np.array(
    [
        [0.00000, 0.07885, 0.03333, 0.88781],
        [0.90585, 0.00000, 0.00000, 0.09415],
        [0.00000, 0.09681, 0.82433, 0.07885],
        [0.06667, 0.67615, 0.22385, 0.03333],
        [0.00000, 0.00000, 0.06397, 0.93603],
        [0.00000, 0.03063, 0.26667, 0.70270],
        [0.72748, 0.03333, 0.06352, 0.17567],
        [0.00000, 0.00000, 0.88781, 0.11219],
        [0.00000, 0.76082, 0.16033, 0.07885],
        [0.00000, 0.82429, 0.06352, 0.11219],
    ]
)
