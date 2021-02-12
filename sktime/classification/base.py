# -*- coding: utf-8 -*-
__all__ = [
    "BaseClassifier",
    "classifier_list",
]
__author__ = ["Markus Löning"]

import numpy as np

from sktime.base import BaseEstimator
from sktime.utils.validation.panel import check_X

"""
Main list of classifiers extending this class. For clarity, some utility classifiers,
such as Proximity Stump, are not listed.
"""
classifier_list = [
    # in classification/distance_based
    "ProximityForest",
    # "KNeighborsTimeSeriesClassifier",
    # "ElasticEnsemble",
    # "ShapeDTW",
    # in classification/dictionary_based
    "BOSS",
    "ContractableBOSS",
    "TemporalDictionaryEnsemble",
    "WEASEL",
    "MUSE",
    # in classification/interval_based
    "RandomIntervalSpectralForest",
    "TimeSeriesForest",
    "CanonicalIntervalForest",
    # in classification/shapelet_based
    "ShapeletTransformClassifier",
    "ROCKET",
    "MrSEQLClassifier",
]


class BaseClassifier(BaseEstimator):
    """
    Base class for classifiers, for identification.
    """

    def fit(self, X, y):
        raise NotImplementedError("abstract method")

    def predict_proba(self, X):
        raise NotImplementedError("abstract method")

    def predict(self, X):
        """
        classify instances
        ----
        Parameters
        ----
        X : panda dataframe
            instances of the dataset
        ----
        Returns
        ----
        predictions : 1d numpy array
            array of predictions of each instance (class value)
        """
        X = check_X(X)
        self.check_is_fitted()
        distributions = self.predict_proba(X)
        predictions = []
        for instance_index in range(0, X.shape[0]):
            distribution = distributions[instance_index]
            prediction = np.argmax(distribution)
            predictions.append(prediction)
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), normalize=True)
