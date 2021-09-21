#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecasters for combining forecasts via stacking."""

__author__ = ["Markus Löning"]
__all__ = ["StackingForecaster"]

from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.forecasting.model_selection import SingleWindowSplitter
from sktime.utils.validation.forecasting import check_regressor


class StackingForecaster(_HeterogenousEnsembleForecaster):
    """StackingForecaster.

    Stacks two or more Forecasters and uses a meta-model (regressor) to infer
    the final predictions from the predictions of the given forecasters.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Estimators to apply to the input series.
    regressor: sklearn-like regressor, optional, default=None.
        The regressor is used as a meta-model and trained with the predictions
        of the ensemble forecasters as exog data and with y as endog data. The
        length of the data is dependent to the given fh. If None, then
        a GradientBoostingRegressor(max_depth=5) is used.
        The regressor can also be a sklearn.Pipeline().
    random_state : int, RandomState instance or None, default=None
        Used to set random_state of the default regressor.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.

    Attributes
    ----------
    regressor_ : sklearn-like regressor
        Fitted meta-model (regressor)

    Examples
    --------
    >>> from sktime.forecasting.compose import StackingForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecasters = [
    ...     ("trend", PolynomialTrendForecaster()),
    ...     ("naive", NaiveForecaster()),
    ... ]
    >>> forecaster = StackingForecaster(forecasters=forecasters)
    >>> forecaster.fit(y=y, fh=[1,2,3])
    StackingForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _required_parameters = ["forecasters"]
    _tags = {
        "univariate-only": False,
        "requires-fh-in-fit": True,
        "handles-missing-data": False,
    }

    def __init__(self, forecasters, regressor=None, random_state=None, n_jobs=None):
        super(StackingForecaster, self).__init__(forecasters=forecasters, n_jobs=n_jobs)
        self.regressor = regressor
        self.random_state = random_state

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

        _, forecasters = self._check_forecasters()
        self.regressor_ = check_regressor(
            regressor=self.regressor, random_state=self.random_state
        )

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
        self.regressor_.fit(X_meta, y_meta)

        # refit forecasters on entire training series
        self._fit_forecasters(forecasters, y, fh=self.fh, X=X)

        return self

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

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
        y_pred = self.regressor_.predict(y_preds)
        index = self.fh.to_absolute(self.cutoff)
        return pd.Series(y_pred, index=index)
