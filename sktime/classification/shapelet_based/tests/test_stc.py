# -*- coding: utf-8 -*-
"""ShapeletTransformClassifier test code."""
import numpy as np
from sklearn.metrics import accuracy_score

from sktime._contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.datasets import load_unit_test


def test_stc_train_estimate():
    """Test of ShapeletTransformClassifier train estimate on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train STC
    stc = ShapeletTransformClassifier(
        estimator=RotationForest(n_estimators=2),
        max_shapelets=3,
        n_shapelet_samples=10,
        batch_size=5,
        random_state=0,
        save_transformed_data=True,
    )
    stc.fit(X_train, y_train)

    # test train estimate
    train_probas = stc._get_train_probs(X_train, y_train)
    assert train_probas.shape == (20, 2)
    train_preds = stc.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.6


def test_contracted_stc():
    """Test of contracted ShapeletTransformClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted STC
    stc = ShapeletTransformClassifier(
        estimator=RotationForest(contract_max_n_estimators=2),
        max_shapelets=3,
        time_limit_in_minutes=0.25,
        contract_max_n_shapelet_samples=10,
        batch_size=5,
        random_state=0,
    )
    stc.fit(X_train, y_train)
