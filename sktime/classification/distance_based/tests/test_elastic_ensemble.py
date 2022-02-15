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
        #        majority_vote=True,
        #        distance_measures=["dtw"],
    )
    ee.fit(X_train, y_train)

    # assert probabilities are the same
    probas = ee.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, ee_unit_test_probas, decimal=4)


ee_unit_test_probas = np.array(
    [
        [0.08130, 0.91870],
        [1.00000, 0.00000],
        [0.08130, 0.91870],
        [1.00000, 0.00000],
        [0.55285, 0.44715],
        [1.00000, 0.00000],
        [0.86179, 0.13821],
        [0.08130, 0.91870],
        [1.00000, 0.00000],
        [1.00000, 0.00000],
    ]
)
