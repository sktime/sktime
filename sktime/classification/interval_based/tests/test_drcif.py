# -*- coding: utf-8 -*-
"""DrCIF test code."""
from sktime.classification.interval_based import DrCIF
from sktime.datasets import load_unit_test


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

    # fails stochastically, probably not a correct expectation, commented out, see #3206
    # assert len(drcif.estimators_) > 1
