#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = [
    "Detrender"
]
__author__ = ["Markus Löning"]

import pandas as pd
from sklearn.base import clone
from sktime.forecasting.base import MetaForecasterMixin
from sktime.transformers.detrend._base import BaseSeriesToSeriesTransformer
from sktime.utils.validation.forecasting import check_y


class Detrender(MetaForecasterMixin, BaseSeriesToSeriesTransformer):

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


class RegressionDetrender(MetaForecasterMixin, BaseSeriesToSeriesTransformer):

    def __init__(self, regressor):
        self.regressor = regressor
        super(RegressionDetrender, self).__init__()

    def fit(self, y, *fit_kwargs):
        y = check_y(y)
        x = self._get_x(y)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(x, y.values)
        self._is_fitted = True
        return self

    def transform(self, y, **transform_kwargs):
        self._check_is_fitted()
        y = check_y(y)
        x = self._get_x(y)
        yt = y - self.regressor_.predict(x)
        return pd.Series(yt, index=y.index)

    def inverse_transform(self, y, **inverse_transform_kwargs):
        self._check_is_fitted()
        y = check_y(y)
        x = self._get_x(y)
        yit = y + self.regressor_.predict(x)
        return pd.Series(yit, index=y.index)

    @staticmethod
    def _get_x(y):
        return y.index.values.reshape(-1, 1)
