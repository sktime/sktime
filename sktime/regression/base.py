#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["BaseRegressor"]

from sktime.base import BaseEstimator


class BaseRegressor(BaseEstimator):
    """
    Base class for regressors, for identification.
    """
    _estimator_type = "regressor"

    def fit(self, X_train, y_train):
        raise NotImplementedError("abstract method")

    def predict(self, X_test):
        raise NotImplementedError("abstract method")

    def score(self, X_test, y_test):
        from sklearn.metrics import r2_score
        return r2_score(y_test, self.predict(X_test))
