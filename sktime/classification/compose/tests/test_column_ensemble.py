# -*- coding: utf-8 -*-
"""Column ensemble test code."""
__author__ = ["TonyBagnall"]


import numpy as np
from numpy import testing

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.interval_based import DrCIF
from sktime.datasets import load_basic_motions, load_unit_test


def test_col_ens_on_basic_motions():
    """Test of ColumnEnsembleClassifier on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
    drcif = DrCIF(n_estimators=2, n_intervals=2, att_subsample_size=2, random_state=0)
    estimators = [
        ("DrCIF", drcif, [0]),
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
    drcif = DrCIF(n_estimators=2, n_intervals=2, att_subsample_size=2, random_state=0)
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
        [0.50000, 0.50000],
        [1.00000, 0.00000],
        [0.50000, 0.50000],
        [1.00000, 0.00000],
        [0.50000, 0.50000],
        [0.50000, 0.50000],
        [0.50000, 0.50000],
        [1.00000, 0.00000],
    ]
)
col_ens_basic_motions_probas = np.array(
    [
        [0.50000, 0.00000, 0.00000, 0.50000],
        [0.50000, 0.00000, 0.00000, 0.50000],
        [0.00000, 0.00000, 1.00000, 0.00000],
        [0.00000, 1.00000, 0.00000, 0.00000],
        [0.00000, 0.00000, 0.00000, 1.00000],
        [0.00000, 0.00000, 0.50000, 0.50000],
        [0.50000, 0.00000, 0.50000, 0.00000],
        [0.50000, 0.00000, 0.50000, 0.00000],
        [0.00000, 1.00000, 0.00000, 0.00000],
        [0.00000, 1.00000, 0.00000, 0.00000],
    ]
)
