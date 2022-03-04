# -*- coding: utf-8 -*-
"""TEASER test code."""
import numpy as np

from sktime.classification.early_classification._teaser import TEASER
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_unit_test
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy


def test_teaser_on_unit_test_data():
    """Test of TEASER on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, y_test = load_unit_test(split="test", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train probability threshold
    pt = TEASER(
        random_state=0,
        classification_points=[6, 10, 16, 24],
        estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
    )
    pt.fit(X_train, y_train)

    X_test = from_nested_to_3d_numpy(X_test)
    states = None
    for i in pt.classification_points:
        X = X_test[indices, :, :i]
        probas = pt.predict_proba(X)
        decisions, states = pt.decide_prediction_safety(X, probas, states)
