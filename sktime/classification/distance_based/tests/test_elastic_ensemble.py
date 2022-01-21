# -*- coding: utf-8 -*-
"""ElasticEnsemble test code."""
import numpy as np
from numpy import testing

from sktime.classification.distance_based import ElasticEnsemble
from sktime.datasets import load_unit_test


def test_ee_on_unit_test_data():
    """Test of ElasticEnsemble on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train EE
    ee = ElasticEnsemble(
        proportion_of_param_options=0.1,
        proportion_train_for_test=0.1,
        random_state=0,
    )
    ee.fit(X_train, y_train)

    # assert probabilities are the same
    probas = ee.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, ee_unit_test_probas, decimal=2)


ee_unit_test_probas = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.1512605, 0.8487395],
        [0.8487395, 0.1512605],
        [0.55462185, 0.44537815],
        [0.8487395, 0.1512605],
        [0.70588235, 0.29411765],
        [0.1512605, 0.8487395],
        [0.8487395, 0.1512605],
        [1.0, 0.0],
    ]
)
