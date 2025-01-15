"""Rotation Forest test code."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sktime.classification.sklearn import RotationForest
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(RotationForest),
    reason="run test if softdeps are present and incrementally (if requested)",
)
@parametrize_with_checks([RotationForest()])
def test_sklearn_compatible_estimator(estimator, check):
    """Run sklearn estimator compatibility checks."""
    check(estimator)


@pytest.mark.skipif(
    not run_test_for_class(RotationForest),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_contracted_rotf():
    """Test of RotF contracting and train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")

    rotf = RotationForest(
        contract_max_n_estimators=5,
        time_limit_in_minutes=0.25,
        save_transformed_data=True,
        random_state=0,
    )
    rotf.fit(X_train, y_train)

    assert len(rotf.estimators_) > 1

    # test train estimate
    train_proba = rotf._get_train_probs(X_train, y_train)
    assert isinstance(train_proba, np.ndarray)
    assert train_proba.shape == (len(X_train), 2)
    np.testing.assert_almost_equal(train_proba.sum(axis=1), 1, decimal=4)
