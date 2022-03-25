# -*- coding: utf-8 -*-
"""Arsenal test code."""
import numpy as np
from sklearn.metrics import accuracy_score

from sktime.classification.kernel_based import Arsenal
from sktime.datasets import load_unit_test


def test_arsenal_train_estimate():
    """Test of Arsenal train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train Arsenal
    arsenal = Arsenal(
        num_kernels=20, n_estimators=5, random_state=0, save_transformed_data=True
    )
    arsenal.fit(X_train, y_train)

    # test train estimate
    train_probas = arsenal._get_train_probs(X_train, y_train)
    assert train_probas.shape == (20, 2)
    train_preds = arsenal.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.6


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
