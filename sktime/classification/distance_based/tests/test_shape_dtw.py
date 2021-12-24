# -*- coding: utf-8 -*-
"""ShapeDTW test code."""
import numpy as np
from numpy import testing

from sktime.classification.distance_based import ShapeDTW
from sktime.datasets import load_unit_test


def test_shapedtw_on_unit_test_data():
    """Test of ShapeDTW on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train ShapeDTW
    shapedtw = ShapeDTW()
    shapedtw.fit(X_train, y_train)

    # assert probabilities are the same
    probas = shapedtw.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, shapedtw_unit_test_probas, decimal=2)


shapedtw_unit_test_probas = np.array(
    [
        [
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
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
    ]
)
