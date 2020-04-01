#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

from sktime.base import BaseEstimator


class BaseSeriesToSeriesTransformer(BaseEstimator):

    def __init__(self):
        super(BaseSeriesToSeriesTransformer, self).__init__()

    def fit(self, y_train, **fit_params):
        self._is_fitted = True
        return self

    def fit_transform(self, y_train, **fit_params):
        """Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        y : pd.Series

        Returns
        -------
        yt : pd.Series
            Transformed time series.
        """
        return self.fit(y_train, **fit_params).transform(y_train)

    def transform(self, y, **transform_params):
        raise NotImplementedError("abstract method")

    def inverse_transform(self, y, **transform_params):
        raise NotImplementedError("abstract method")

    def update(self, y_new, update_params=False):
        raise NotImplementedError("abstract method")
