"""Test function of DummyClassifier."""

import numpy as np
import pytest

from sktime.classification.dummy import DummyClassifier
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(DummyClassifier),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_dummy_classifier():
    """Test function for DummyClassifier."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy3D")
    X_test, _ = load_unit_test(split="test", return_type="numpy3D")
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    pred = dummy.predict(X_test)
    assert all(i == "1" for i in pred)
    pred_proba = dummy.predict_proba(X_test)
    assert all(np.array_equal([0.5, 0.5], i) for i in pred_proba)
