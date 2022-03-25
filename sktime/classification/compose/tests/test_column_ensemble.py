# -*- coding: utf-8 -*-
"""Column ensemble test code."""
__author__ = ["TonyBagnall"]


import numpy as np
from numpy import testing

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.datasets import load_basic_motions


def test_col_ens_on_basic_motions():
    """Test of ColumnEnsembleClassifier on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    cboss = ContractableBOSS(n_parameter_samples=4, max_ensemble_size=2, random_state=0)
    cif = CanonicalIntervalForest(
        n_estimators=2, n_intervals=4, att_subsample_size=4, random_state=0
    )
    estimators = [
        ("cBOSS", cboss, [5]),
        ("CIF", cif, [3, 4]),
    ]

    # train column ensemble
    col_ens = ColumnEnsembleClassifier(estimators=estimators)
    col_ens.fit(X_train.iloc[indices], y_train[indices])

    probas = col_ens.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, col_ens_basic_motions_probas, decimal=2)


col_ens_basic_motions_probas = np.array(
    [
        [0.0, 0.0, 0.25, 0.75],
        [0.25, 0.0, 0.3324742268041237, 0.4175257731958763],
        [0.0, 0.0, 0.3324742268041237, 0.6675257731958764],
        [0.5, 0.0, 0.08247422680412371, 0.4175257731958763],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5],
        [0.25, 0.25, 0.08247422680412371, 0.4175257731958763],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.5, 0.08247422680412371, 0.4175257731958763],
        [0.0, 0.25, 0.08247422680412371, 0.6675257731958764],
    ]
)
