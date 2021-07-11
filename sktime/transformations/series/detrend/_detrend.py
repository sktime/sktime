#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["Detrender"]
__author__ = ["Markus Löning", "Svea Meyer"]

from sklearn.base import clone
import pandas as pd

from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series
from sktime.forecasting.trend import PolynomialTrendForecaster


class Detrender(_SeriesToSeriesTransformer):
    """
    Remove a trend from a series.
    This transformer uses any forecaster and returns the in-sample residuals
    of the forecaster's predicted values.

    The Detrender works by first fitting the forecaster to the input data.
    To transform data, it uses the fitted forecaster to generate
    forecasts for the time points of the passed data and returns the residuals
     of the forecasts.
    Depending on the passed data, this will require it to generate in-sample
    or out-of-sample forecasts.

    The detrender also works in a pipeline as a form of boosting,
    by first detrending a time series and then fitting another forecaster on
    the residuals.

    For example, to remove the linear trend of a time series:
    forecaster = PolynomialTrendForecaster(degree=1)
    transformer = Detrender(forecaster=forecaster)
    yt = transformer.fit_transform(y_train)

    Parameters
    ----------
    forecaster : estimator object, optional
        default=None. If None, PolynomialTrendForecaster(degree=1) is used.

        The forecasting model to remove the trend with
        (e.g. PolynomialTrendForecaster)

    Attributes
    ----------
    forecaster_ : estimator object
        Model that defines the trend in the series

    Example
    ----------
    >>> from sktime.transformations.series.detrend import Detrender
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Detrender(forecaster=PolynomialTrendForecaster(degree=1))
    >>> y_hat = transformer.fit_transform(y)
    """

    _required_parameters = ["forecaster"]
    _tags = {"transform-returns-same-time-index": True}

    def __init__(self, forecaster=None):
        self.forecaster = forecaster
        self.forecaster_ = None
        super(Detrender, self).__init__()

    def fit(self, Z, X=None):
        """
        Compute the trend in the series

        Parameters
        ----------
        Y : pd.Series
            Endogenous time series to fit a trend to.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables

        Returns
        -------
        self : an instance of self
        """
        self._is_fitted = False
        z = check_series(Z)
        if self.forecaster is None:
            self.forecaster = PolynomialTrendForecaster(degree=1)

        # multivariate
        if isinstance(z, pd.DataFrame):
            self.forecaster_ = {}
            for colname in z.columns:
                forecaster = clone(self.forecaster)
                self.forecaster_[colname] = forecaster.fit(z[colname], X)
        # univariate
        else:
            forecaster = clone(self.forecaster)
            self.forecaster_ = forecaster.fit(z, X)
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """
        Remove trend from the data.

        Parameters
        ----------
        y : pd.Series
            Time series to be detrended
        X : pd.DataFrame, optional (default=False)
            Exogenous variables

        Returns
        -------
        y_hat : pd.Series
            De-trended series
        """
        self.check_is_fitted()
        z = check_series(Z)
        fh = ForecastingHorizon(z.index, is_relative=False)

        # multivariate
        if isinstance(z, pd.DataFrame):
            z = z.copy()
            # check if all columns are known
            Z_fit_keys = set(self.forecaster_.keys())
            Z_new_keys = set(z.columns)
            difference = Z_new_keys.difference(Z_fit_keys)
            if len(difference) != 0:
                raise ValueError(
                    "Z contains columns that have not been "
                    "seen in fit: " + str(difference)
                )
            for colname in z.columns:
                z_pred = self.forecaster_[colname].predict(fh, X)
                z[colname] = z[colname] - z_pred
            return z
        # univariate
        else:
            z_pred = self.forecaster_.predict(fh, X)
            return z - z_pred

    def inverse_transform(self, Z, X=None):
        """
        Add trend back to a time series

        Parameters
        ----------
        y : pd.Series, list
            Detrended time series to revert
        X : pd.DataFrame, optional (default=False)
            Exogenous variables

        Returns
        -------
        y_hat : pd.Series
            Series with the trend
        """
        self.check_is_fitted()
        z = check_series(Z)
        fh = ForecastingHorizon(z.index, is_relative=False)

        # multivariate
        if isinstance(z, pd.DataFrame):
            z = z.copy()
            # check if all columns are known
            Z_fit_keys = set(self.forecaster_.keys())
            Z_new_keys = set(z.columns)
            difference = Z_new_keys.difference(Z_fit_keys)
            if len(difference) != 0:
                raise ValueError(
                    "Z contains columns that have not been "
                    "seen in fit: " + difference
                )
            for colname in z.columns:
                z_pred = self.forecaster_[colname].predict(fh, X)
                z[colname] = z[colname] + z_pred
            return z
        # univariate
        else:
            z_pred = self.forecaster_.predict(fh, X)
            return z + z_pred

    def update(self, Z, X=None, update_params=True):
        """
        Update the parameters of the detrending estimator with new data

        Parameters
        ----------
        y_new : pd.Series
            New time series
        update_params : bool, optional (default=True)
            Update the parameters of the detrender model with

        Returns
        -------
        self : an instance of self
        """
        z = check_series(Z, allow_empty=True)
        # multivariate
        if isinstance(z, pd.DataFrame):
            # check if all columns are known
            Z_fit_keys = set(self.forecaster_.keys())
            Z_new_keys = set(z.columns)
            difference = Z_new_keys.difference(Z_fit_keys)
            if len(difference) != 0:
                raise ValueError(
                    "Z contains columns that have not been "
                    "seen in fit: " + str(difference)
                )
            for colname in z.columns:
                self.forecaster_[colname].update(
                    z[colname], X, update_params=update_params
                )
        # univariate
        else:
            self.forecaster_.update(z, X, update_params=update_params)
        return self
