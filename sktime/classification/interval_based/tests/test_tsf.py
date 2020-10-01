# -*- coding: utf-8 -*-
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sktime.classification.interval_based import TimeSeriesForest
from sktime.datasets import load_gunpoint


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_on_gunpoint(n_jobs):
    np.random.seed(42)
    X, y = load_gunpoint(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    tsf = TimeSeriesForest(random_state=42, n_estimators=20, n_jobs=n_jobs)
    tsf.fit(X_train, y_train)

    n_classes = len(np.unique(y))
    proba = tsf.predict_proba(X_test)

    tsf_score = tsf.score(X_test, y_test)

    assert tsf_score >= 0.95, "Classifier performs worse than expected"
