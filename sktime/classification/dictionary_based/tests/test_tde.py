# -*- coding: utf-8 -*-
"""TDE test code."""
from sktime.classification.dictionary_based._tde import TemporalDictionaryEnsemble
from sktime.datasets import load_unit_test


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
