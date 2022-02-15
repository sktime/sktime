# -*- coding: utf-8 -*-
"""ProximityForest test code."""
import numpy as np
from numpy import testing

from sktime.classification.distance_based import ProximityForest
from sktime.datasets import load_unit_test


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


pf_unit_test_probas = np.array(
    [
        [0.60000, 0.40000],
        [0.80000, 0.20000],
        [0.00000, 1.00000],
        [0.80000, 0.20000],
        [0.20000, 0.80000],
        [1.00000, 0.00000],
        [0.80000, 0.20000],
        [0.00000, 1.00000],
        [0.80000, 0.20000],
        [1.00000, 0.00000],
    ]
)
