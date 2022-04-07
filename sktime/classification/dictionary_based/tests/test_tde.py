# -*- coding: utf-8 -*-
"""TDE test code."""
import numpy as np
from sklearn.metrics import accuracy_score

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
        save_train_predictions=True,
    )
    tde.fit(X_train, y_train)

    # test loocv train estimate
    train_probas = tde._get_train_probs(X_train, y_train)
    assert train_probas.shape == (20, 2)
    train_preds = tde.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.6

    # test oob estimate
    train_probas = tde._get_train_probs(X_train, y_train, train_estimate_method="oob")
    assert train_probas.shape == (20, 2)
    train_preds = tde.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.6


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

    assert len(tde.estimators_) > 1
