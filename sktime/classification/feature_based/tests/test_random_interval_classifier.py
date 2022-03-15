# -*- coding: utf-8 -*-
"""RandomIntervalClassifier test code."""
import numpy as np
from numpy import testing
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.feature_based import RandomIntervalClassifier
from sktime.datasets import load_basic_motions, load_unit_test
from sktime.transformations.series.summarize import SummaryTransformer


def test_random_interval_classifier_on_unit_test_data():
    """Test of RandomIntervalClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train random interval classifier
    ric = RandomIntervalClassifier(
        random_state=0,
        n_intervals=5,
        interval_transformers=SummaryTransformer(
            summary_function=("mean", "std", "min", "max"),
            quantiles=(0.25, 0.5, 0.75),
        ),
        estimator=RandomForestClassifier(n_estimators=10),
    )
    ric.fit(X_train, y_train)

    # assert probabilities are the same
    probas = ric.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, random_interval_classifier_unit_test_probas, decimal=2
    )


def test_random_interval_classifier_on_basic_motions():
    """Test of RandomIntervalClassifier on basic motions."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train random interval classifier
    ric = RandomIntervalClassifier(
        random_state=0,
        n_intervals=5,
        interval_transformers=SummaryTransformer(
            summary_function=("mean", "std", "min", "max"),
            quantiles=(0.25, 0.5, 0.75),
        ),
        estimator=RandomForestClassifier(n_estimators=10),
    )
    ric.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = ric.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(
        probas, random_interval_classifier_basic_motions_probas, decimal=2
    )


random_interval_classifier_unit_test_probas = np.array(
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
            0.9,
            0.1,
        ],
        [
            0.2,
            0.8,
        ],
        [
            0.9,
            0.1,
        ],
        [
            0.9,
            0.1,
        ],
    ]
)
random_interval_classifier_basic_motions_probas = np.array(
    [
        [
            0.0,
            0.0,
            0.2,
            0.8,
        ],
        [
            0.2,
            0.3,
            0.1,
            0.4,
        ],
        [
            0.0,
            0.0,
            0.8,
            0.2,
        ],
        [
            0.2,
            0.6,
            0.0,
            0.2,
        ],
        [
            0.0,
            0.0,
            0.2,
            0.8,
        ],
        [
            0.0,
            0.1,
            0.5,
            0.4,
        ],
        [
            0.3,
            0.2,
            0.1,
            0.4,
        ],
        [
            0.0,
            0.0,
            0.9,
            0.1,
        ],
        [
            0.0,
            0.9,
            0.0,
            0.1,
        ],
        [
            0.2,
            0.8,
            0.0,
            0.0,
        ],
    ]
)
