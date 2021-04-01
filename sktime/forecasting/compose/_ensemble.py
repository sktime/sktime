#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["EnsembleForecaster"]

import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin


class EnsembleForecaster(
    _OptionalForecastingHorizonMixin, _HeterogenousEnsembleForecaster
):
    """Ensemble of forecasters

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    agg : {'mean', 'median', 'min', 'max'} (default='mean')
    """

    _required_parameters = ["forecasters"]

    def __init__(self, forecasters, n_jobs=None, agg="mean"):
        super(EnsembleForecaster, self).__init__(forecasters=forecasters, n_jobs=n_jobs)
        self.agg = agg

    def fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self._set_y_X(y, X)
        self._set_fh(fh)
        names, forecasters = self._check_forecasters()
        self._fit_forecasters(forecasters, y, X, fh)
        self._is_fitted = True
        return self

    def update(self, y, X=None, update_params=True):
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
        self.check_is_fitted()
        self._update_y_X(y, X)
        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """reduce the prediction

        Parameters
        ----------
        fh : int, list or np.array
        X : pd.DataFrame, optional (default=None)
        return_pred_int : boolean (default= False)
        alpha : float (default= DEFAULT_ALPHA)

        Returns
        -------
        reduction : concatenated dataframe of aggregate values
        """
        if return_pred_int:
            raise NotImplementedError()
        if self.agg not in ("mean", "median", "min", "max"):
            raise ValueError(
                "Invalid aggregate passed. Valid choices are: 'mean', 'median', 'min', 'max'"
            )
        if self.agg == "median":
            return pd.concat(self._predict_forecasters(fh, X), axis=1).median(axis=1)
        if self.agg == "min":
            return pd.concat(self._predict_forecasters(fh, X), axis=1).min(axis=1)
        if self.agg == "max":
            return pd.concat(self._predict_forecasters(fh, X), axis=1).max(axis=1)
        return pd.concat(self._predict_forecasters(fh, X), axis=1).mean(axis=1)
