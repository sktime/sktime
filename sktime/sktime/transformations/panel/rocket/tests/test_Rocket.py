"""Rocket test code."""

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.datasets import load_gunpoint
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.rocket import Rocket


@pytest.mark.skipif(
    not run_test_for_class(Rocket),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_rocket_on_gunpoint():
    """Test of Rocket on gun point."""
    # load training data
    X_training, Y_training = load_gunpoint(split="train", return_X_y=True)

    # 'fit' ROCKET -> infer data dimensions, generate random kernels
    ROCKET = Rocket(num_kernels=10_000, random_state=0)
    ROCKET.fit(X_training)

    # transform training data
    X_training_transform = ROCKET.transform(X_training)

    # test shape of transformed training data -> (number of training
    # examples, num_kernels * 2)
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 20_000))

    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # load test data
    X_test, Y_test = load_gunpoint(split="test", return_X_y=True)

    # transform test data
    X_test_transform = ROCKET.transform(X_test)

    # test shape of transformed test data -> (number of test examples,
    # num_kernels * 2)
    np.testing.assert_equal(X_test_transform.shape, (len(X_test), 20_000))

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)

    # test predictions (on Gunpoint, should be 100% accurate)
    assert accuracy == 1.0
