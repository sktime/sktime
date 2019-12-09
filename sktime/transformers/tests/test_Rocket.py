import numpy as np

from sktime.datasets import load_gunpoint
from sktime.transformers.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV

def test_rocket_on_gunpoint():

    # load training data
    X_training, Y_training = load_gunpoint(split = "TRAIN", return_X_y = True)

    # 'fit' ROCKET -> infer data dimensions, generate random kernels
    ROCKET = Rocket(num_kernels = 10_000)
    ROCKET.fit(X_training)

    # transform training data
    X_training_transform = ROCKET.transform(X_training)

    # fit classifier
    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    classifier.fit(X_training_transform, Y_training)

    # load test data
    X_test, Y_test = load_gunpoint(split = "TEST", return_X_y = True)

    # transform test data
    X_test_transform = ROCKET.transform(X_test)

    # predict
    classifier.score(X_test_transform, Y_test)
