# -*- coding: utf-8 -*-
"""MiniRocketMultivariateVariable test code."""
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.datasets import load_japanese_vowels
from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable


def test_minirocket_multivariate_variable_on_japanese_vowels():
    """Test of MiniRocketMultivariate on japanese vowels."""
    # load training data
    X_training, Y_training = load_japanese_vowels(split="train", return_X_y=True)

    # 'fit' MINIROCKET -> infer data dimensions, generate random kernels
    num_kernels = 10_000
    minirocket_mv_var = MiniRocketMultivariateVariable(
        num_kernels=num_kernels, 
        add_padding_short_series=0,
        reference_length="max",
        max_dilations_per_kernel=32,
        n_jobs=1, 
        random_state=42,
    )
    minirocket_mv_var.fit(X_training)

    # transform training data
    X_training_transform = minirocket_mv_var.transform(X_training)

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 1000)
    np.testing.assert_equal(
        X_training_transform.shape, (len(X_training), 84 * (num_kernels // 84))
    )

    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # load test data
    X_test, Y_test = load_japanese_vowels(split="test", return_X_y=True)

    # transform test data
    X_test_transform = minirocket_mv_var.transform(X_test)

    # test shape of transformed test data -> (number of test examples,
    # nearest multiple of 84 < 10,000)
    np.testing.assert_equal(
        X_test_transform.shape, (len(X_test), 84 * (num_kernels // 84))
    )

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)

    # test accuracy, mean usually .987, and minimum .983
    assert accuracy > 0.975, "Test accuracy should be greater than 0.975"


"""
# exhaustive test of MiniRocketMultivariateVariable on japanese_vowels
def test_minirocket_multivariate_variable_on_japanese_vowels_exaustive():
    X_training, Y_training = load_japanese_vowels(split="train", return_X_y=True)
    X_test, Y_test = load_japanese_vowels(split="test", return_X_y=True)
    accs = []
    for i in range(3):
        classifier = make_pipeline(
            MiniRocketMultivariateVariable(random_state=i),
            StandardScaler(with_mean=False),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        classifier.fit(X_training, Y_training)
        accs.append(
            classifier.score(X_test, Y_test)
        )
    assert np.median(accs) > 0.98, "Test accuracy should be greater than 0.98"
"""