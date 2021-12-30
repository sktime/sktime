# -*- coding: utf-8 -*-
"""RandomIntervalSpectralEnsemble test code."""
import numpy as np
from numpy import testing

from sktime.classification.interval_based import RandomIntervalSpectralEnsemble
from sktime.datasets import load_unit_test


def test_rise_on_unit_test_data():
    """Test of RandomIntervalSpectralEnsemble on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train RISE
    rise = RandomIntervalSpectralEnsemble(n_estimators=10, random_state=0)
    rise.fit(X_train, y_train)

    # assert probabilities are the same
    probas = rise.predict_proba(X_test.iloc[indices])
    testing.assert_array_equal(probas, rise_unit_test_probas)


rise_unit_test_probas = np.array(
    [
        [
            0.1,
            0.9,
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
            0.7,
            0.3,
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
            0.6,
            0.4,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.7,
            0.3,
        ],
        [
            0.9,
            0.1,
        ],
    ]
)
