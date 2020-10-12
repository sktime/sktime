#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["BaseRegressor"]

from sktime.base import BaseEstimator


class BaseRegressor(BaseEstimator):
    """
    Base class for regressors, for identification.
    """

    def fit(self, X, y):
        raise NotImplementedError("abstract method")

    def predict(self, X):
        raise NotImplementedError("abstract method")

    def score(self, X, y):
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))
