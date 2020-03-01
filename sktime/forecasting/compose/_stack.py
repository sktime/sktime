#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["StackingForecaster"]

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.base import is_regressor
from sktime.forecasting._base import BaseHeterogenousMetaForecaster
from sktime.forecasting._base import DEFAULT_ALPHA
from sktime.forecasting._base import OptionalForecastingHorizonMixin


class StackingForecaster(OptionalForecastingHorizonMixin, BaseHeterogenousMetaForecaster):
    _required_parameters = ["forecasters", "final_regressor"]

    def __init__(self, forecasters, final_regressor, n_jobs=None):
        self.final_regressor = final_regressor
        self.final_regressor_ = None
        self.n_jobs = n_jobs
        super(StackingForecaster, self).__init__(forecasters=forecasters)

    def fit(self, y_train, fh=None, X_train=None):
        self._set_oh(y_train)
        self._set_fh(fh)

        names, forecasters = self._check_forecasters()
        self._check_final_regressor()

        # fit forecasters
        self._fit_forecasters(forecasters, y_train, X_train)

        # make in-sample prediction
        fh = -np.arange(len(y_train))
        y_pred_ins = self._predict(fh, X=X_train)

        # fit final regressor on in-sample predictions
        self.final_regressor_ = clone(self.final_regressor)
        self.final_regressor_.fit(y_pred_ins, y_train)

        self._is_fitted = True
        return self

    def update(self, y_new, X_new=None, update_params=False):
        self._check_is_fitted()
        self._set_oh(y_new)
        if update_params:
            warn("Updating `final regressor is not implemented")
        for forecaster in self.forecasters_:
            forecaster.update(y_new, X_new=X_new, update_params=update_params)
        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        self._check_is_fitted()
        self._set_fh(fh)
        if return_pred_int:
            raise NotImplementedError()
        y_preds = self._predict(self.fh, X=X)
        y_pred = self.final_regressor_.predict(y_preds)
        index = self._get_absolute_fh(self.fh)
        return pd.Series(y_pred, index=index)

    def _predict(self, fh=None, X=None):
        return np.column_stack(self._predict_forecasters(fh, X=X))

    def _check_final_regressor(self):
        if not is_regressor(self.final_regressor):
            raise ValueError(f"`'`final_regressor` should be a regressor, but found: {self.final_regressor}")
