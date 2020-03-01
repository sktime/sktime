#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

from sklearn.base import BaseEstimator
from sktime.utils.exceptions import NotFittedError


class BaseSeriesToSeriesTransformer(BaseEstimator):

    def __init__(self):
        self._is_fitted = False

    @property
    def is_fitted(self):
        """Has `fit` been called?"""
        return self._is_fitted

    def _check_is_fitted(self):
        """Check if the forecaster has been fitted.

        Raises
        ------
        NotFittedError
            if the forecaster has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(f"This instance of {self.__class__.__name__} has not "
                                 f"been fitted yet; please call `fit` first.")

    def fit(self, y, **fit_params):
        self._is_fitted = True
        return self

    def fit_transform(self, y, **fit_params):
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
        return self.fit(y, **fit_params).transform(y)

    def transform(self, y, **transform_params):
        raise NotImplementedError("abstract method")
