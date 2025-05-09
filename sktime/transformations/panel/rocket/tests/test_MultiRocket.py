"""MultiRocket test code."""

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.datasets import load_gunpoint
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.rocket import MultiRocket


@pytest.mark.skipif(
    not run_test_for_class(MultiRocket),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multirocket_on_gunpoint():
    """Test of MultiRocket on gun point."""
    # load training data
    X_training, Y_training = load_gunpoint(split="train", return_X_y=True)

    # 'fit' MultiRocket -> infer data dimensions, generate random kernels
    multirocket = MultiRocket(random_state=0)
    multirocket.fit(X_training)

    # transform training data
    X_training_transform = multirocket.transform(X_training)

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 4*84=336 < 50,000 (2*4*6_250))
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 49_728))

    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # load test data
    X_test, Y_test = load_gunpoint(split="test", return_X_y=True)

    # transform test data
    X_test_transform = multirocket.transform(X_test)

    # test shape of transformed test data -> (number of test examples,
    # nearest multiple of 4*84=336 < 50,000 (2*4*6_250))
    np.testing.assert_equal(X_test_transform.shape, (len(X_test), 49_728))

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)

    # test predictions (on Gunpoint, should be > 99% accurate)
    assert accuracy > 0.99
