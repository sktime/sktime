# -*- coding: utf-8 -*-
"""MultiRocketClassifier test code."""
import numpy as np
from numpy import testing

from sktime.classification.kernel_based import MultiRocketClassifier
from sktime.datasets import load_unit_test


def test_multirocket_on_unit_test_data():
    """Test of MultiRocketClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train MultiRocketClassifier
    multirocket = MultiRocketClassifier(num_kernels=2500, random_state=0)
    multirocket.fit(X_train, y_train)

    # assert probabilities are the same
    probas = multirocket.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, rocket_unit_test_probas, decimal=2)


rocket_unit_test_probas = np.array(
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
