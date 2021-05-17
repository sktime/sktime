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


class AverageMixin:

    @staticmethod
    def average(series, iterations=100):
        """
        Method called to find the average for a distance metric

        Parameters
        ----------
        series:
            The set of sequences to average

        iterations: int default = 100
            The number of iterations
        """
        raise NotImplementedError("abstract method")
