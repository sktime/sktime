# -*- coding: utf-8 -*-
"""RocketClassifier test code."""
import numpy as np
from numpy import testing

from sktime.classification.kernel_based import RocketClassifier
from sktime.datasets import load_basic_motions, load_unit_test


def test_rocket_on_unit_test_data():
    """Test of RocketClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train Rocket
    rocket = RocketClassifier(num_kernels=500, random_state=0)
    rocket.fit(X_train, y_train)

    # assert probabilities are the same
    probas = rocket.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, rocket_unit_test_probas, decimal=2)


def test_rocket_on_basic_motions():
    """Test of RocketClassifier on basic motions."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train Rocket
    rocket = RocketClassifier(num_kernels=500, random_state=0)
    rocket.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = rocket.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, rocket_basic_motions_probas, decimal=2)


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
rocket_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            0.0,
            0.0,
        ],
    ]
)
