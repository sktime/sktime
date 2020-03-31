#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = [
    "PolynomialTrendForecaster"
]

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sktime.forecasting.base._sktime import OptionalForecastingHorizonMixin, BaseSktimeForecaster, DEFAULT_ALPHA


class PolynomialTrendForecaster(OptionalForecastingHorizonMixin, BaseSktimeForecaster):

    def __init__(self, regressor=None, degree=1, with_intercept=True):
        self.regressor = regressor
        self.degree = degree
        self.with_intercept = with_intercept
        self.regressor_ = None
        super(PolynomialTrendForecaster, self).__init__()

    def fit(self, y_train, fh=None, X_train=None):
        if X_train is not None:
            raise NotImplementedError()
        self._set_oh(y_train)
        self._set_fh(fh)

        # for default regressor, set fit_intercept=False as we generate a dummy variable in polynomial features
        r = self.regressor if self.regressor is not None else LinearRegression(fit_intercept=False)  #
        self.regressor_ = make_pipeline(PolynomialFeatures(degree=self.degree, include_bias=self.with_intercept), r)
        x = y_train.index.values.reshape(-1, 1)
        self.regressor_.fit(x, y_train.values)
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int or X is not None:
            raise NotImplementedError()
        self.check_is_fitted()
        self._set_fh(fh)
        fh_abs = self.fh.absolute(self.cutoff)
        x = fh_abs.reshape(-1, 1)
        y_pred = self.regressor_.predict(x)
        return pd.Series(y_pred, index=fh_abs)
