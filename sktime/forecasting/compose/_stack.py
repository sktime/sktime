#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["StackingForecaster"]

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.base import is_regressor

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.forecasting.model_selection import SingleWindowSplitter

from warnings import warn


class StackingForecaster(_HeterogenousEnsembleForecaster):
    """StackingForecaster.

    Stacks two or more Forecasters

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
    final_regressor: Regressor
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    """

    _required_parameters = ["forecasters", "final_regressor"]
    _tags = {
        "univariate-only": True,
        "requires-fh-in-fit": True,
        "handles-missing-data": False,
    }

    def __init__(self, forecasters, final_regressor, n_jobs=None):
        super(StackingForecaster, self).__init__(forecasters=forecasters, n_jobs=n_jobs)
        self.final_regressor = final_regressor
        self.final_regressor_ = None

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        if X is not None:
            raise NotImplementedError()

        names, forecasters = self._check_forecasters()
        self._check_final_regressor()

        # split training series into training set to fit forecasters and
        # validation set to fit meta-learner
        cv = SingleWindowSplitter(fh=self.fh.to_relative(self.cutoff))
        train_window, test_window = next(cv.split(y))
        y_fcst = y.iloc[train_window]
        y_meta = y.iloc[test_window].values

        # fit forecasters on training window
        self._fit_forecasters(forecasters, y_fcst, fh=self.fh, X=X)
        X_meta = np.column_stack(self._predict_forecasters(X))

        # fit final regressor on on validation window
        self.final_regressor_ = clone(self.final_regressor)
        self.final_regressor_.fit(X_meta, y_meta)

        # refit forecasters on entire training series
        self._fit_forecasters(forecasters, y, fh=self.fh, X=X)

        return self

    def _update(self, y, X=None, update_params=True):

        """Update fitted parameters

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        if update_params:
            warn("Updating `final regressor is not implemented")
        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)
        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        if return_pred_int:
            raise NotImplementedError()
        y_preds = np.column_stack(self._predict_forecasters(X))
        y_pred = self.final_regressor_.predict(y_preds)
        index = self.fh.to_absolute(self.cutoff)
        return pd.Series(y_pred, index=index)

    def _check_final_regressor(self):
        if not is_regressor(self.final_regressor):
            raise ValueError(
                f"`final_regressor` should be a regressor, "
                f"but found: {self.final_regressor}"
            )
