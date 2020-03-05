#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["EnsembleForecaster"]

import pandas as pd
from sktime.forecasting.base import BaseHeterogenousEnsembleForecaster, OptionalForecastingHorizonMixin
from sktime.forecasting.base import DEFAULT_ALPHA


class EnsembleForecaster(OptionalForecastingHorizonMixin, BaseHeterogenousEnsembleForecaster):

    _required_parameters = ["forecasters"]

    def __init__(self, forecasters, n_jobs=None):
        self.n_jobs = n_jobs
        super(EnsembleForecaster, self).__init__(forecasters=forecasters)

    def fit(self, y_train, fh=None, X_train=None):
        self._set_oh(y_train)
        self._set_fh(fh)
        names, forecasters = self._check_forecasters()
        self._fit_forecasters(forecasters, y_train, fh=fh, X_train=X_train)
        self._is_fitted = True
        return self

    def update(self, y_new, X_new=None, update_params=False):
        self._check_is_fitted()
        self._set_oh(y_new)
        for forecaster in self.forecasters_:
            forecaster.update(y_new, X_new=X_new, update_params=update_params)
        return self

    def transform(self, fh=None, X=None):
        self._check_is_fitted()
        self._set_fh(fh)
        return pd.concat(self._predict_forecasters(fh=self.fh, X=X), axis=1)

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()
        return pd.concat(self._predict_forecasters(self.fh, X=X), axis=1).mean(axis=1)
