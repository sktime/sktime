# Base class for the Keras neural networks adapted from Fawaz et. al
# https://github.com/hfawaz/dl-4-tsc
#
# @article{fawaz2019deep,
#   title={Deep learning for time series classification: a review},
#   author={Fawaz, Hassan Ismail and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
#   journal={Data Mining and Knowledge Discovery},
#   pages={1--47},
#   year={2019},
#   publisher={Springer}
# }
#
# File also contains some simple unit-esque tests for the networks and
# their compatibility wit hthe rest of the package,
# and experiments for confirming accurate reproduction
#
# todo proper unit tests

__author__ = "James Large, Aaron Bostrom"

import numpy as np
import pandas as pd

from sktime.classifiers.base import BaseClassifier
from sktime.classifiers.tests.test_dl4tscnetworks import comparisonExperiments
from sktime.classifiers.tests.test_dl4tscnetworks import test_all_networks_all_tests

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class BaseDeepLearner(BaseClassifier):
    classes_ = None
    nb_classes = None

    def build_model(self, input_shape, nb_classes, **kwargs):
        raise NotImplementedError('this is an abstract method')

    def fit(self, X, y, input_checks=True, **kwargs):
        raise NotImplementedError()

    def predict_proba(self, X, input_checks=True, **kwargs):

        if isinstance(X, pd.DataFrame):
            if isinstance(X.iloc[0, self.dim_to_use], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:, 0]])
            else:
                raise TypeError(
                    "Input should either be a 2d numpy array, or a pandas dataframe containing Series objects")

        if len(X.shape) == 2:
            # add a dimension to make it multivariate with one dimension
            X = X.reshape((X.shape[0], X.shape[1], 1))

        probs = self.model.predict(X, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def convert_y(self, y):
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        # categories='auto' to get rid of FutureWarning

        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.nb_classes = len(self.classes_)

        y = y.reshape(len(y), 1)
        y = self.onehot_encoder.fit_transform(y)

        return y


if __name__ == "__main__":
    comparisonExperiments()
    # test_all_networks_all_tests()
