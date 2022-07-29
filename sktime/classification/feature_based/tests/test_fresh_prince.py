# -*- coding: utf-8 -*-
"""FreshPRINCE test code."""
import numpy as np
from sklearn.metrics import accuracy_score

from sktime.classification.feature_based import FreshPRINCE
from sktime.datasets import load_unit_test


def test_fresh_prince_train_estimate():
    """Test of FreshPRINCE train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train FreshPRINCE classifier
    fp = FreshPRINCE(
        n_estimators=2,
        default_fc_parameters="minimal",
        random_state=0,
        save_transformed_data=True,
    )
    fp.fit(X_train, y_train)

    # test train estimate
    train_probas = fp._get_train_probs(X_train, y_train)
    assert train_probas.shape == (20, 2)
    train_preds = fp.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.6
