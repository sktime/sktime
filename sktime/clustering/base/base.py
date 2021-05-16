# -*- coding: utf-8 -*-
from sktime.base import BaseEstimator


class ClusterMixin(BaseEstimator):

    def fit(self, X):
        raise NotImplementedError("abstract method")

    def predict(self, X):
        raise NotImplementedError("abstract method")


class TransformerMixin:

    def fit_transform(self, X, y=None, sample_weight=None):
        raise NotImplementedError("abstract method")
