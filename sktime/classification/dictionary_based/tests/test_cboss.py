# -*- coding: utf-8 -*-
"""cBOSS test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.dictionary_based import ContractableBOSS
from sktime.datasets import load_unit_test


def test_cboss_on_unit_test_data():
    """Test of cBOSS on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train cBOSS
    cboss = ContractableBOSS(
        n_parameter_samples=10,
        max_ensemble_size=5,
        random_state=0,
        save_train_predictions=True,
    )
    cboss.fit(X_train, y_train)

    # assert probabilities are the same
    probas = cboss.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, cboss_unit_test_probas, decimal=2)

    # test train estimate
    train_probas = cboss._get_train_probs(X_train, y_train)
    train_preds = cboss.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


cboss_unit_test_probas = np.array(
    [
        [0.11264679623652221, 0.8873532037634777],
        [0.6030149028226083, 0.3969850971773916],
        [0.0, 1.0],
        [0.7747064075269555, 0.22529359247304442],
        [0.6620596112904332, 0.3379403887095666],
        [0.7747064075269555, 0.22529359247304442],
        [0.28433830094086937, 0.7156616990591305],
        [0.0, 1.0],
        [0.1716915047043472, 0.8283084952956528],
        [0.6030149028226083, 0.3969850971773916],
    ]
)
