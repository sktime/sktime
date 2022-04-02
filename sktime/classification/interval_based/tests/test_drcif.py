# -*- coding: utf-8 -*-
"""DrCIF test code."""
import numpy as np
from sklearn.metrics import accuracy_score

from sktime.classification.interval_based import DrCIF
from sktime.datasets import load_unit_test


def test_drcif_train_estimate():
    """Test of DrCIF on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train DrCIF
    drcif = DrCIF(
        n_estimators=2,
        n_intervals=2,
        att_subsample_size=2,
        random_state=0,
        save_transformed_data=True,
    )
    drcif.fit(X_train, y_train)

    # test train estimate
    train_probas = drcif._get_train_probs(X_train, y_train)
    assert train_probas.shape == (20, 2)
    train_preds = drcif.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.6


def test_contracted_drcif():
    """Test of contracted DrCIF on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted DrCIF
    drcif = DrCIF(
        time_limit_in_minutes=0.25,
        contract_max_n_estimators=2,
        n_intervals=2,
        att_subsample_size=2,
        random_state=0,
    )
    drcif.fit(X_train, y_train)

    assert len(drcif.estimators_) > 1
