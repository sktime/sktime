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

    assert proba.shape == (
        X_test.shape[0],
        n_classes,
    ), "Incorrect shape of classes probabilities"
    np.testing.assert_array_equal(
        np.ones(X_test.shape[0]),
        np.sum(proba, axis=1),
        "Incorrect classes probabilities",
    )

    tsf_score = tsf.score(X_test, y_test)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    dummy_score = dummy.score(X_test, y_test)

    assert tsf_score > dummy_score, "Classifier performs worse than dummy classifier"

    assert tsf_score >= 0.95, "Classifier performs worse than expected"


def test_multiproc_determinism():
    np.random.seed(42)
    X, y = load_gunpoint(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    probas = []
    for n_jobs in [1, 4]:
        tsf = TimeSeriesForest(random_state=42, n_estimators=20, n_jobs=n_jobs)
        tsf.fit(X_train, y_train)

        probas.append(tsf.predict_proba(X_test))

    np.testing.assert_array_almost_equal(
        probas[0],
        probas[1],
        err_msg="Probabilites for test set not equal between 1 and 4 job run",
    )
