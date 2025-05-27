import pytest
from sktime.classification.feature_based import TSFreshClassifier
from sktime.datasets import load_unit_test
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def test_tsfresh_predict_proba_feature_order_consistency():
    """Check that TSFreshClassifier.predict_proba works with consistent feature ordering."""
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    X_test, _ = load_unit_test(split="test", return_X_y=True)

    clf = TSFreshClassifier(estimator=RandomForestClassifier(n_estimators=5, random_state=42), n_jobs=1)
    clf.fit(X_train, y_train)

    try:
        y_proba = clf.predict_proba(X_test)
    except ValueError as e:
        pytest.fail(f"predict_proba failed due to feature mismatch: {e}")

    # Basic sanity check
    assert isinstance(y_proba, np.ndarray)
    assert y_proba.shape[0] == len(X_test)