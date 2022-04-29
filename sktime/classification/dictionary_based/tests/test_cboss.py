# -*- coding: utf-8 -*-
"""cBOSS test code."""
import numpy as np
from sklearn.metrics import accuracy_score

from sktime.classification.dictionary_based import ContractableBOSS
from sktime.datasets import load_unit_test


def test_cboss_train_estimate():
    """Test of cBOSS train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train cBOSS
    cboss = ContractableBOSS(
        n_parameter_samples=4,
        max_ensemble_size=2,
        random_state=0,
        save_train_predictions=True,
    )
    cboss.fit(X_train, y_train)

    # test train estimate
    train_probas = cboss._get_train_probs(X_train, y_train)
    assert train_probas.shape == (20, 2)
    train_preds = cboss.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.6
