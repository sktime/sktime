# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_gunpoint

# expected y_proba
expected = np.array(
    [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.95, 0.05],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.95, 0.05],
        [0.0, 1.0],
        [0.95, 0.05],
        [1.0, 0.0],
        [0.15, 0.85],
        [0.9, 0.1],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.2, 0.8],
        [1.0, 0.0],
        [0.9, 0.1],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
)


def test_y_proba_on_gunpoint():
    X, y = load_gunpoint(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    estimator = TimeSeriesForestClassifier(random_state=42, n_estimators=20)
    estimator.fit(X_train, y_train)
    actual = estimator.predict_proba(X_test)
    np.testing.assert_array_equal(actual, expected)
