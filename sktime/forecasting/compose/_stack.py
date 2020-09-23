#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["StackingForecaster"]

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.base import is_regressor
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import BaseHeterogenousEnsembleForecaster
from sktime.forecasting.base._sktime import RequiredForecastingHorizonMixin
from sktime.forecasting.model_selection import SingleWindowSplitter


class StackingForecaster(RequiredForecastingHorizonMixin,
                         BaseHeterogenousEnsembleForecaster):
    _required_parameters = ["forecasters", "final_regressor"]

    def __init__(self, forecasters, final_regressor, n_jobs=None):
        self.final_regressor = final_regressor
        self.final_regressor_ = None
        self.n_jobs = n_jobs
        super(StackingForecaster, self).__init__(forecasters=forecasters)

    def fit(self, y_train, fh=None, X_train=None):
        """Fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X_train : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self._set_y_X(y_train, X_train)
        if X_train is not None:
            raise NotImplementedError()
        self._set_fh(fh)

        names, forecasters = self._check_forecasters()
        self._check_final_regressor()

        # split training series into training set to fit forecasters and
        # validation set to fit meta-learner
        cv = SingleWindowSplitter(fh=self.fh)
        training_window, test_window = next(cv.split(y_train))
        y_fcst = y_train.iloc[training_window]
        y_meta = y_train.iloc[test_window].values

        # fit forecasters on training window
        self._fit_forecasters(forecasters, y_fcst, fh=self.fh, X_train=X_train)
        X_meta = np.column_stack(self._predict_forecasters(X=X_train))

        # fit final regressor on on validation window
        self.final_regressor_ = clone(self.final_regressor)
        self.final_regressor_.fit(X_meta, y_meta)

        # refit forecasters on entire training series
        self._fit_forecasters(forecasters, y_train, fh=self.fh,
                              X_train=X_train)

        self._is_fitted = True
        return self

    def update(self, y_new, X_new=None, update_params=False):
        """Update fitted parameters

        Parameters
        ----------
        y_new : pd.Series
        X_new : pd.DataFrame
        update_params : bool, optional (default=False)

        Returns
        -------
        self : an instance of self
        """
        self.check_is_fitted()
        self._update_y_X(y_new, X_new)
        if update_params:
            warn("Updating `final regressor is not implemented")
        for forecaster in self.forecasters_:
            forecaster.update(y_new, X_new=X_new, update_params=update_params)
        return self

    def _predict(self, fh=None, X=None, return_pred_int=False,
                 alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()
        y_preds = np.column_stack(self._predict_forecasters(X=X))
        y_pred = self.final_regressor_.predict(y_preds)
        index = self.fh.absolute(self.cutoff)
        return pd.Series(y_pred, index=index)

    def _check_final_regressor(self):
        if not is_regressor(self.final_regressor):
            raise ValueError(f"`final_regressor` should be a regressor, "
                             f"but found: {self.final_regressor}")
