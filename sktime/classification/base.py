# -*- coding: utf-8 -*-
__all__ = ["BaseClassifier"]
__author__ = ["Markus Löning"]

from sktime.base import BaseEstimator

from sktime.utils import comparison
from sktime.utils.validation.panel import check_X


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
            prediction = comparison.arg_max(distribution, self.random_state)
            predictions.append(prediction)
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions

    def score(self, X, y):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), normalize=True)
