# -*- coding: utf-8 -*-
"""SupervisedTimeSeriesForest test code."""
import numpy as np
from numpy import testing

from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.datasets import load_unit_test


def test_stsf_on_unit_test_data():
    """Test of SupervisedTimeSeriesForest on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train STSF
    stsf = SupervisedTimeSeriesForest(n_estimators=10, random_state=0)
    stsf.fit(X_train, y_train)

    # assert probabilities are the same
    probas = stsf.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, stsf_unit_test_probas)


stsf_unit_test_probas = np.array(
    [
        [
            0.0,
            1.0,
        ],
        [
            0.8,
            0.2,
        ],
        [
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
            1.0,
            0.0,
        ],
        [
            1.0,
            0.0,
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
            1.0,
            0.0,
        ],
    ]
)
