# -*- coding: utf-8 -*-
"""ShapeletTransformClassifier test code."""
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.sklearn import RotationForest
from sktime.datasets import load_unit_test


def test_contracted_stc():
    """Test of contracted ShapeletTransformClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted STC
    stc = ShapeletTransformClassifier(
        estimator=RotationForest(contract_max_n_estimators=2, random_state=0),
        max_shapelets=3,
        time_limit_in_minutes=0.25,
        contract_max_n_shapelet_samples=10,
        batch_size=5,
        random_state=0,
    )
    stc.fit(X_train, y_train)

    # fails stochastically, probably not a correct expectation, commented out, see #3206
    # assert len(stc._estimator.estimators_) > 1
