#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["PolynomialTrendForecaster"]

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.utils.datetime import _get_duration


class PolynomialTrendForecaster(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    """
    Forecast time series data with a polynomial trend.
    Default settings train a linear regression model with a 1st degree
    polynomial transformation of the feature.

    Parameters
    ----------
    regressor : estimator object, optional (default = None)
        Define the regression model type. If not set, will default to
         sklearn.linear_model.LinearRegression
    degree : int, optional (default = 1)
        Degree of polynomial function
    with_intercept : bool, optional (default=True)
        If true, then include a feature in which all polynomial powers are
        zero. (i.e. a column of ones, acts as an intercept term in a linear
        model)

    Attributes
    ----------

    """

    def __init__(self, regressor=None, degree=1, with_intercept=True):
        self.regressor = regressor
        self.degree = degree
        self.with_intercept = with_intercept
        self.regressor_ = None
        super(PolynomialTrendForecaster, self).__init__()

    def fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series with which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecast horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        if X is not None:
            raise NotImplementedError("Exogeneous variables are not " "yet supported")
        self._set_y_X(y, X)
        self._set_fh(fh)

        # for default regressor, set fit_intercept=False as we generate a
        # dummy variable in polynomial features
        if self.regressor is None:
            regressor = LinearRegression(fit_intercept=False)
        else:
            regressor = self.regressor

        # make pipeline with polynomial features
        self.regressor_ = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=self.with_intercept),
            regressor,
        )

        # transform data
        n_timepoints = _get_duration(self._y.index, coerce_to_int=True) + 1
        X = np.arange(n_timepoints).reshape(-1, 1)

        # fit regressor
        self.regressor_.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Make forecasts for the given forecast horizon

        Parameters
        ----------
        fh : int, list or np.array
            The forecast horizon with the steps ahead to predict
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored)
        return_pred_int : bool, optional (default=False)
            Return the prediction intervals for the forecast.
        alpha : float or list, optional (default=0.95)
            If alpha is iterable, multiple intervals will be calculated.

        Returns
        -------
        y_pred : pd.Series
            Point predictions for the forecast
        y_pred_int : pd.DataFrame
            Prediction intervals for the forecast
        """
        if return_pred_int or X is not None:
            raise NotImplementedError()
        self.check_is_fitted()
        self._set_fh(fh)

        # use relative fh as time index to predict
        fh = self.fh.to_absolute_int(self._y.index[0], self.cutoff)
        X_pred = fh.to_numpy().reshape(-1, 1)
        y_pred = self.regressor_.predict(X_pred)
        return pd.Series(y_pred, index=self.fh.to_absolute(self.cutoff))
