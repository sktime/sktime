"""Tests for TSFreshClassifier."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.feature_based import TSFreshClassifier
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TSFreshClassifier),
    reason="test only if soft dependencies are present",
)
def test_tsfresh_predict_proba_feature_order_consistency():
    """Check that TSFreshClassifier.predict_proba has consistent feature ordering."""
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, _ = load_unit_test(split="test", return_X_y=True)

    sk_clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf = TSFreshClassifier(estimator=sk_clf, n_jobs=1)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)

    # Basic sanity check
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape[0] == len(X_test)
