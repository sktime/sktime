# -*- coding: utf-8 -*-
"""Column ensemble test code."""
__author__ = ["TonyBagnall"]


import numpy as np
from numpy import testing

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.datasets import load_basic_motions, load_unit_test


def test_col_ens_on_unit_test_data():
    """Test of ColumnEnsembleClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train Column ensemble with a single
    col_ens = ColumnEnsembleClassifier()
    col_ens.fit(X_train, y_train)
    preds = col_ens.predict(X_test.iloc[indices])
    assert preds[0] == 1
    # assert probabilities are the same
    probas = col_ens.predict_proba(X_test.iloc[indices])

    testing.assert_array_almost_equal(probas, col_ens_unit_test_probas, decimal=2)


def test_col_ens_on_basic_motions():
    """Test of ColumnEnsembleClassifier on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train column ensemble
    col_ens = ColumnEnsembleClassifier()
    col_ens.fit(X_train.iloc[indices], y_train[indices])
    preds = col_ens.predict(X_test.iloc[indices])
    assert preds[0] == 1
    # assert probabilities are the same
    probas = col_ens.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, col_ens_unit_test_probas, decimal=2)


col_ens_unit_test_probas = np.array(
    [
        [
            0.0,
            1.0,
        ],
        [
            0.49241837193506105,
            0.5075816280649389,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.9043327688966699,
            0.09566723110333018,
        ],
        [
            0.8016244295841345,
            0.19837557041586543,
        ],
        [
            1.0,
            0.0,
        ],
        [
            0.7059571984808044,
            0.2940428015191956,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.8016244295841345,
            0.19837557041586543,
        ],
        [
            1.0,
            0.0,
        ],
    ]
)
col_ens_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            0.6261191124951343,
            0.3738808875048657,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.7477617750097314,
            0.0,
            0.0,
            0.25223822499026854,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            0.7477617750097314,
            0.25223822499026854,
            0.0,
        ],
        [
            0.0,
            0.7477617750097314,
            0.25223822499026854,
            0.0,
        ],
    ]
)
