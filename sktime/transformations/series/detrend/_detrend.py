#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformations to detrend a time series."""

__all__ = ["Detrender"]
__author__ = ["mloning", "SveaMeyer13", "KishManani"]

import pandas as pd

from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.base import BaseTransformer


class Detrender(BaseTransformer):
    """Remove a :term:`trend <Trend>` from a series.

    This transformer uses any forecaster and returns the in-sample residuals
    of the forecaster's predicted values.

    The Detrender works as follows:
    in "fit", the forecaster is fit to the input data.
    in "transform", the forecast residuals are computed and return.
    Depending on time indices, this can generate in-sample or out-of-sample residuals.

    For example, to remove the linear trend of a time series:
        forecaster = PolynomialTrendForecaster(degree=1)
        transformer = Detrender(forecaster=forecaster)
        yt = transformer.fit_transform(y_train)

    The detrender can also be used in a pipeline for residual boosting,
    by first detrending and then fitting another forecaster on residuals.

    Parameters
    ----------
    forecaster : sktime forecaster, follows BaseForecaster, default = None.
        The forecasting model to remove the trend with
            (e.g. PolynomialTrendForecaster).
        If forecaster is None, PolynomialTrendForecaster(degree=1) is used.
    model : {"additive", "multiplicative"}, default="additive"
        If `model="additive"` the `forecaster` is fit to the original time
        series and the `transform` method subtracts the trend from the time series.
        If `model="multiplicative"` the `forecaster` is fit to the original time
        series and the `transform` method divides the trend from the time series.

    Attributes
    ----------
    forecaster_ : Fitted forecaster
        Forecaster that defines the trend in the series.

    See Also
    --------
    Deseasonalizer
    STLTransformer

    Examples
    --------
    >>> from sktime.transformations.series.detrend import Detrender
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Detrender(forecaster=PolynomialTrendForecaster(degree=1))
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for y?
        "univariate-only": True,
        "fit_is_empty": False,
        "capability:inverse_transform": True,
        "transform-returns-same-time-index": True,
    }

    def __init__(self, forecaster=None, model="additive"):
        self.forecaster = forecaster
        self.model = model
        self.forecaster_ = None
        super(Detrender, self).__init__()

        # whether this transformer is univariate depends on the forecaster
        #  this transformer is univariate iff the forecaster is univariate
        if forecaster is not None:
            fc_univ = forecaster.get_tag("scitype:y", "univariate") == "univariate"
            self.set_tags(**{"univariate-only": fc_univ})

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to fit transform to
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        if self.forecaster is None:
            self.forecaster = PolynomialTrendForecaster(degree=1)

        # univariate: X is pd.Series
        if isinstance(X, pd.Series):
            forecaster = self.forecaster.clone()
            # note: the y in the transformer is exogeneous in the forecaster, i.e., X
            self.forecaster_ = forecaster.fit(y=X, X=y)
        # multivariate
        elif isinstance(X, pd.DataFrame):
            self.forecaster_ = {}
            for colname in X.columns:
                forecaster = self.forecaster.clone()
                self.forecaster_[colname] = forecaster.fit(y=X[colname], X=y)
        else:
            raise TypeError("X must be pd.Series or pd.DataFrame")

        allowed_models = ("additive", "multiplicative")
        if self.model not in allowed_models:
            raise ValueError("`model` must be 'additive' or 'multiplicative'")

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            transformed version of X, detrended series
        """
        fh = ForecastingHorizon(X.index, is_relative=False)

        # univariate: X is pd.Series
        if isinstance(X, pd.Series):
            # note: the y in the transformer is exogeneous in the forecaster, i.e., X
            X_pred = self.forecaster_.predict(fh=fh, X=y)
            if self.model == "additive":
                return X - X_pred
            elif self.model == "multiplicative":
                return X / X_pred
        # multivariate: X is pd.DataFrame
        elif isinstance(X, pd.DataFrame):
            Xt = X.copy()
            # check if all columns are known
            X_fit_keys = set(self.forecaster_.keys())
            X_new_keys = set(X.columns)
            difference = X_new_keys.difference(X_fit_keys)
            if len(difference) != 0:
                raise ValueError(
                    "X contains columns that have not been "
                    "seen in fit: " + str(difference)
                )
            for colname in Xt.columns:
                X_pred = self.forecaster_[colname].predict(fh=fh, X=y)
                if self.model == "additive":
                    Xt[colname] = Xt[colname] - X_pred
                elif self.model == "multiplicative":
                    Xt[colname] = Xt[colname] / X_pred
            return Xt
        else:
            raise TypeError("X must be pd.Series or pd.DataFrame")

    def _inverse_transform(self, X, y=None):
        """Logic used by `inverse_transform` to reverse transformation on `X`.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be inverse transformed
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            inverse transformed version of X
        """
        fh = ForecastingHorizon(X.index, is_relative=False)

        # univariate: X is pd.Series
        if isinstance(X, pd.Series):
            # note: the y in the transformer is exogeneous in the forecaster, i.e., X
            X_pred = self.forecaster_.predict(fh=fh, X=y)
            if self.model == "additive":
                return X + X_pred
            elif self.model == "multiplicative":
                return X * X_pred
        # multivariate: X is pd.DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            # check if all columns are known
            X_fit_keys = set(self.forecaster_.keys())
            X_new_keys = set(X.columns)
            difference = X_new_keys.difference(X_fit_keys)
            if len(difference) != 0:
                raise ValueError(
                    "X contains columns that have not been "
                    "seen in fit: " + str(difference)
                )
            for colname in X.columns:
                X_pred = self.forecaster_[colname].predict(fh=fh, X=y)
                if self.model == "additive":
                    X[colname] = X[colname] + X_pred
                elif self.model == "multiplicative":
                    X[colname] = X[colname] * X_pred

            return X

    def _update(self, X, y=None, update_params=True):
        """Update the parameters of the detrending estimator with new data.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to fit transform to
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation
        update_params : bool, default=True
            whether the model is updated. Yes if true, if false, simply skips call.
            argument exists for compatibility with forecasting module.

        Returns
        -------
        self : an instance of self
        """
        # multivariate
        if isinstance(X, pd.DataFrame):
            # check if all columns are known
            X_fit_keys = set(self.forecaster_.keys())
            X_new_keys = set(X.columns)
            difference = X_new_keys.difference(X_fit_keys)
            if len(difference) != 0:
                raise ValueError(
                    "Z contains columns that have not been "
                    "seen in fit: " + str(difference)
                )
            for colname in X.columns:
                self.forecaster_[colname].update(
                    y=X[colname], X=y, update_params=update_params
                )
        # univariate
        else:
            self.forecaster_.update(y=X, X=y, update_params=update_params)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.trend import TrendForecaster

        return {"forecaster": TrendForecaster()}
