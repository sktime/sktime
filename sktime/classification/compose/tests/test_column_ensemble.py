# -*- coding: utf-8 -*-
"""Column ensemble test code."""
__author__ = ["TonyBagnall"]


import numpy as np
from numpy import testing

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.interval_based import DrCIF
from sktime.datasets import load_basic_motions, load_unit_test


def test_col_ens_on_basic_motions():
    """Test of ColumnEnsembleClassifier on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=10,
        max_ensemble_size=5,
        randomly_selected_params=5,
        random_state=0,
    )
    drcif = DrCIF(n_estimators=10, random_state=0)
    estimators = [
        ("TDE", tde, [3, 4]),
        ("DrCIF", drcif, [5]),
    ]

    # train column ensemble
    col_ens = ColumnEnsembleClassifier(estimators=estimators)
    col_ens.fit(X_train, y_train)
    probas = col_ens.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, col_ens_basic_motions_probas, decimal=2)


def test_col_ens_on_unit_test_data():
    """Test of ColumnEnsembleClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
    drcif = DrCIF(n_estimators=10, random_state=0)
    estimators = [("DrCIF", drcif, [0])]
    col_ens = ColumnEnsembleClassifier(estimators=estimators)
    col_ens.fit(X_train, y_train)
    # assert probabilities are the same
    probas = col_ens.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, col_ens_unit_test_probas, decimal=2)


col_ens_unit_test_probas = np.array(
    [
        [0.00000, 1.00000],
        [1.00000, 0.00000],
        [0.00000, 1.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [0.90000, 0.10000],
        [0.00000, 1.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
    ]
)

col_ens_basic_motions_probas = np.array(
    [
        [0.00000, 0.11828, 0.00000, 0.88172],
        [0.85878, 0.00000, 0.00000, 0.14122],
        [0.00000, 0.14522, 0.73650, 0.11828],
        [0.10000, 0.51423, 0.33577, 0.05000],
        [0.00000, 0.00000, 0.04595, 0.95405],
        [0.00000, 0.04595, 0.10000, 0.85405],
        [0.59122, 0.05000, 0.09528, 0.26350],
        [0.00000, 0.00000, 0.83172, 0.16828],
        [0.00000, 0.64122, 0.24049, 0.11828],
        [0.00000, 0.78644, 0.09528, 0.11828],
    ]
)
