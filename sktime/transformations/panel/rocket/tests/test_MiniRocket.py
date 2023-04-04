# -*- coding: utf-8 -*-
"""MiniRocket test code."""
import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.datasets import load_gunpoint
from sktime.transformations.panel.rocket import MiniRocket
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_minirocket_on_gunpoint():
    """Test of MiniRocket on gun point."""
    # load training data
    X_training, Y_training = load_gunpoint(split="train", return_X_y=True)

    # 'fit' MINIROCKET -> infer data dimensions, generate random kernels
    minirocket = MiniRocket(random_state=0)
    minirocket.fit(X_training)

    # transform training data
    X_training_transform = minirocket.transform(X_training)

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 9_996))

    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # load test data
    X_test, Y_test = load_gunpoint(split="test", return_X_y=True)

    # transform test data
    X_test_transform = minirocket.transform(X_test)

    # test shape of transformed test data -> (number of test examples,
    # nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_test_transform.shape, (len(X_test), 9_996))

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)

    # test predictions (on Gunpoint, should be > 99% accurate)
    assert accuracy > 0.99
