# -*- coding: utf-8 -*-
"""ProbabilityThresholdEarlyClassifier test code."""
import numpy as np
from numpy import testing

from sktime.classification.early_classification import (
    ProbabilityThresholdEarlyClassifier,
)
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_unit_test
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy


def test_prob_threshold_on_unit_test_data():
    """Test of ProbabilityThresholdEarlyClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train probability threshold
    pt = ProbabilityThresholdEarlyClassifier(
        random_state=0,
        classification_points=[6, 16, 24],
        probability_threshold=1,
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
    )
    pt.fit(X_train, y_train)

    final_probas = np.zeros((10, 2))
    final_decisions = np.zeros(10)

    X_test = from_nested_to_3d_numpy(X_test)
    states = None
    for i in pt.classification_points:
        X = X_test[indices, :, :i]
        probas = pt.predict_proba(X)
        decisions, states = pt.decide_prediction_safety(X, probas, states)

        for n in range(10):
            if decisions[n] and final_decisions[n] == 0:
                final_probas[n] = probas[n]
                final_decisions[n] = i

    testing.assert_array_equal(final_probas, pt_unit_test_probas)


pt_unit_test_probas = np.array(
    [
        [
            0.0,
            1.0,
        ],
        [
            0.7,
            0.3,
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
