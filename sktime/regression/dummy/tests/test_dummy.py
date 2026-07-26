"""Test function of DummyRegressor."""

import pytest

from sktime.datasets import load_unit_test
from sktime.regression.dummy import DummyRegressor
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(DummyRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_dummy_regressor():
    """Test function for DummyRegressor."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy3D")
    X_test, _ = load_unit_test(split="test", return_type="numpy3D")
    dummy = DummyRegressor()
    dummy.fit(X_train, y_train)
    pred = dummy.predict(X_test)
    assert (pred == 1.5).all()
