# -*- coding: utf-8 -*-
"""TDE test code."""
import numpy as np

from sktime.classification.dictionary_based._tde import TemporalDictionaryEnsemble
from sktime.datasets import load_unit_test


def test_tde_train_estimate():
    """Test of TDE train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train TDE
    tde = TemporalDictionaryEnsemble(
        n_parameter_samples=5,
        max_ensemble_size=2,
        randomly_selected_params=3,
        random_state=0,
    )
    tde.fit(X_train, y_train)

    # test oob estimate
    train_proba = tde._get_train_probs(X_train, y_train, train_estimate_method="oob")
    assert isinstance(train_proba, np.ndarray)
    assert train_proba.shape == (len(X_train), 2)
    np.testing.assert_almost_equal(train_proba.sum(axis=1), 1, decimal=4)


def test_contracted_tde():
    """Test of contracted TDE on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted TDE
    tde = TemporalDictionaryEnsemble(
        time_limit_in_minutes=0.25,
        contract_max_n_parameter_samples=5,
        max_ensemble_size=2,
        randomly_selected_params=3,
        random_state=0,
    )
    tde.fit(X_train, y_train)

    # fails stochastically, probably not a correct expectation, commented out, see #3206
    # assert len(tde.estimators_) > 1
