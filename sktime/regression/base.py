#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["BaseRegressor", "is_regressor"]

from sktime.base import BaseEstimator


class BaseRegressor(BaseEstimator):
    """
    Base class for regressors, for identification.
    """

    def fit(self, X_train, y_train):
        raise NotImplementedError("abstract method")

    def predict(self, X_test):
        raise NotImplementedError("abstract method")

    def score(self, X_test, y_test):
        from sklearn.metrics import r2_score
        return r2_score(y_test, self.predict(X_test))


def is_regressor(estimator):
    """Return True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.
    """
    return isinstance(estimator, BaseRegressor)
