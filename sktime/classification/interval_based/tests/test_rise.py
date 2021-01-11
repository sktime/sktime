# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

from sktime.classification.interval_based import RandomIntervalSpectralForest
from sktime.datasets import load_gunpoint

# expected y_proba
expected = np.array(
    [
        [0.8, 0.2],
        [1.0, 0.0],
        [0.8, 0.2],
        [0.9, 0.1],
        [0.05, 0.95],
        [0.95, 0.05],
        [0.2, 0.8],
        [0.8, 0.2],
        [1.0, 0.0],
        [0.05, 0.95],
        [0.75, 0.25],
        [0.95, 0.05],
        [0.95, 0.05],
        [0.35, 0.65],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.95, 0.05],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.1, 0.9],
    ]
)


def test_y_proba_on_gunpoint():
    X, y = load_gunpoint(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    estimator = RandomIntervalSpectralForest(random_state=42, n_estimators=20)
    estimator.fit(X_train, y_train)
    actual = estimator.predict_proba(X_test)
    np.testing.assert_array_equal(actual, expected)
