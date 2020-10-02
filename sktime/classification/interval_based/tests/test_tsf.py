# -*- coding: utf-8 -*-
import numpy as np
import pytest
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

    tsf_score = tsf.score(X_test, y_test)

    assert tsf_score >= 0.95, "Classifier performs worse than expected"
