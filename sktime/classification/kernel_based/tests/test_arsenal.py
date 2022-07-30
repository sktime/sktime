# -*- coding: utf-8 -*-
"""Arsenal test code."""
from sktime.classification.kernel_based import Arsenal
from sktime.datasets import load_unit_test


def test_contracted_arsenal():
    """Test of contracted Arsenal on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted Arsenal
    arsenal = Arsenal(
        time_limit_in_minutes=0.25,
        contract_max_n_estimators=5,
        num_kernels=20,
        random_state=0,
    )
    arsenal.fit(X_train, y_train)

    assert len(arsenal.estimators_) > 1
