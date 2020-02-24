#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = [
    "Detrender"
]
__author__ = ["Markus Löning"]

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sktime.utils.exceptions import NotFittedError
from sktime.utils.validation.forecasting import check_y


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

    def transform(self, y, **transform_args):
        raise NotImplementedError("abstract method")


class Detrender(BaseSeriesToSeriesTransformer):

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.forecaster_ = None
        super(Detrender, self).__init__()

    def fit(self, y_train, X_train=None):
        forecaster = clone(self.forecaster)
        self.forecaster_ = forecaster.fit(y_train, X_train=X_train)
        self._is_fitted = True
        return self

    def transform(self, y, X=None):
        self._check_is_fitted()
        y = check_y(y)
        fh = self._get_relative_fh(y)
        y_pred = self.forecaster_.predict(fh=fh, X=X)
        return y - y_pred

    def inverse_transform(self, y, X=None):
        self._check_is_fitted()
        y = check_y(y)
        fh = self._get_relative_fh(y)
        y_pred = self.forecaster_.predict(fh=fh, X=X)
        return y + y_pred

    def _get_relative_fh(self, y):
        return y.index.values - self.forecaster_.cutoff
