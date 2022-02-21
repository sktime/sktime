# -*- coding: utf-8 -*-
"""ProximityForest test code."""
import numpy as np
from numpy import testing

from sktime.classification.distance_based import (
    ProximityForest,
    ProximityStump,
    ProximityTree,
)
from sktime.datasets import load_unit_test


def test_prox_stump_on_unit_test_data():
    """Test of ProximityStump on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train PF
    ps = ProximityStump(random_state=0)
    ps.fit(X_train, y_train)

    # assert probabilities are the same
    probas = ps.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, ps_unit_test_probas, decimal=2)


def test_prox_tree_on_unit_test_data():
    """Test of ProximityTree on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train PF
    pt = ProximityTree(random_state=0)
    pt.fit(X_train, y_train)

    # assert probabilities are the same
    probas = pt.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, pt_unit_test_probas, decimal=2)


def test_pf_on_unit_test_data():
    """Test of ProximityForest on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train PF
    pf = ProximityForest(n_estimators=5, random_state=0)
    pf.fit(X_train, y_train)

    # assert probabilities are the same
    probas = pf.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, pf_unit_test_probas, decimal=2)


ps_unit_test_probas = np.array(
    [
        [0.61043, 0.38957],
        [0.75875, 0.24125],
        [0.40459, 0.59541],
        [0.37091, 0.62909],
        [0.50074, 0.49926],
        [0.35964, 0.64036],
        [0.41061, 0.58939],
        [0.49457, 0.50543],
        [0.69884, 0.30116],
        [0.41854, 0.58146],
    ]
)
pt_unit_test_probas = np.array(
    [
        [0.00000, 1.00000],
        [1.00000, 0.00000],
        [0.00000, 1.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [0.00000, 1.00000],
        [1.00000, 0.00000],
        [0.00000, 1.00000],
        [0.00000, 1.00000],
        [1.00000, 0.00000],
    ]
)
pf_unit_test_probas = np.array(
    [
        [0.20000, 0.80000],
        [1.00000, 0.00000],
        [0.20000, 0.80000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [0.20000, 0.80000],
        [0.20000, 0.80000],
        [0.40000, 0.60000],
        [1.00000, 0.00000],
    ]
)
