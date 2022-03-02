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
    pt = ProximityTree(random_state=0, n_stump_evaluations=1)
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
    pf = ProximityForest(n_estimators=2, random_state=0, n_stump_evaluations=1)
    pf.fit(X_train, y_train)

    # assert probabilities are the same
    probas = pf.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, pf_unit_test_probas, decimal=2)


ps_unit_test_probas = np.array(
    [
        [0.57352, 0.42648],
        [0.56580, 0.43420],
        [0.50489, 0.49511],
        [0.63456, 0.36544],
        [0.39541, 0.60459],
        [0.58939, 0.41061],
        [0.56392, 0.43608],
        [0.52174, 0.47826],
        [0.57691, 0.42309],
        [0.69668, 0.30332],
    ]
)


pt_unit_test_probas = np.array(
    [
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [0.00000, 1.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
    ]
)

pf_unit_test_probas = np.array(
    [
        [0.50000, 0.50000],
        [1.00000, 0.00000],
        [0.50000, 0.50000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
        [0.50000, 0.50000],
        [0.50000, 0.50000],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
    ]
)
