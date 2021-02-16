#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["Detrender"]
__author__ = ["Markus Löning"]

from sklearn.base import clone

from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


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
    forecaster : estimator object
        The forecasting model to remove the trend with
        (e.g. PolynomialTrendForecaster)

    Attributes
    ----------
    forecaster_ : estimator object
        Model that defines the trend in the series
    """

    _required_parameters = ["forecaster"]
    _tags = {"transform-returns-same-time-index": True, "univariate-only": True}

    def __init__(self, forecaster):
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
        z = check_series(Z, enforce_univariate=True)
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
        z = check_series(Z, enforce_univariate=True)
        fh = ForecastingHorizon(z.index, is_relative=False)
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
        z = check_series(Z, enforce_univariate=True)
        fh = ForecastingHorizon(z.index, is_relative=False)
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
        z = check_series(Z, enforce_univariate=True, allow_empty=True)
        self.forecaster_.update(z, X, update_params=update_params)
        return self
