# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.rocket import MiniRocketMultivariate


def test_minirocket_multivariate_on_basic_motions():

    # load training data
    X_training, Y_training = load_basic_motions(split="train", return_X_y=True)

    # 'fit' MINIROCKET -> infer data dimensions, generate random kernels
    minirocket = MiniRocketMultivariate()
    minirocket.fit(X_training)

    # transform training data
    X_training_transform = minirocket.transform(X_training)

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 9_996))

    # fit classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    classifier.fit(X_training_transform, Y_training)

    # load test data
    X_test, Y_test = load_basic_motions(split="test", return_X_y=True)

    # transform test data
    X_test_transform = minirocket.transform(X_test)

    # test shape of transformed test data -> (number of test examples,
    # nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_test_transform.shape, (len(X_test), 9_996))

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)

    # test predictions (on BasicMotions, should be 100% accurate)
    assert accuracy == 1.0
