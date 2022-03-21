#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Transformer to impute missing values in series."""

__author__ = ["aiwalter", "fkiraly"]
__all__ = ["Imputer"]


import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import check_random_state

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.base import BaseTransformer


class Imputer(BaseTransformer):
    """Missing value imputation.

    The Imputer transforms input series by replacing missing values according
    to an imputation strategy specified by `method`.

    IMPORTANT NOTE: this transformer may lead to information leakage on the test set
        for some choices of method when used for forecasting.
        It is safe for time series classification, or in a panel setting where entire
        series are either in the training or in the test set.
        Leakage-free methods in all settings are:
            "constant", "mean_fit", "median_fit"

    Parameters
    ----------
    method : str, default="drift"
        Method to fill the missing values values.

        * "drift" : drift/trend values by sktime.PolynomialTrendForecaster()
        * "linear" : linear interpolation, uses pd.Series.interpolate()
        * "nearest" : use nearest value, uses pd.Series.interpolate()
        * "constant" : same constant value (given in arg value) for all NaN
        * "mean" : uses mean value of X seen in *transform*
        * "mean_fit" : uses mean value of X seen in *fit*
        * "median" : uses median value of X seen in *transform*
        * "median_fit" : uses median value of X seen in *fit*
        * "backfill" ot "bfill" : uses pd.Series.fillna()
        * "pad" or "ffill" : uses pd.Series.fillna()
        * "random" : random values between pd.Series.min() and .max()
            if pd.Series dtype is int, sample is uniform discrete
            if pd.Series dtype is float, sample is uniform continuous
        * "forecaster" : use an sktime Forecaster, given in arg forecaster
            first, X in *transform* is filled with ffill then bfill
            then forecaster is fitted to filled X, and predict values are queried
            at indices which had missing values

    missing_values : int/float/str, default=None
        The placeholder for the missing values. All occurrences of
        missing_values will be imputed, in addition to np.nan.
        If None, then only np.nan values are imputed.
    value : int/float, default=None
        Value to use to fill missing values when method="constant".
    forecaster : Any Forecaster based on sktime.BaseForecaster, default=None
        Use a given Forecaster to impute by insample predictions when
        method="forecaster". Before fitting, missing data is imputed with
            method="ffill" or "bfill" as heuristic.
        if X is multivariate, forecaster must be able to handle multivariate data
            (otherwise use ColumnEnsembleForecaster to wrap a univariate forecaster)
    random_state : int/float/str, optional
        Value to set random.seed() if method="random", default None

    Examples
    --------
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Imputer(method="drift")
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
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": True,
        "handles-missing-data": True,
        "skip-inverse-transform": True,
        "univariate-only": False,
    }

    ALLOWED_METHODS = [
        "drift",
        "linear",
        "nearest",
        "constant",
        "mean",
        "mean_fit",
        "median",
        "median_fit",
        "backfill",
        "bfill",
        "pad",
        "ffill",
        "random",
        "forecaster",
    ]

    METHODS_REQUIRE_FIT = ["mean_fit", "median_fit"]

    def __init__(
        self,
        method="drift",
        random_state=None,
        value=None,
        forecaster=None,
        missing_values=None,
    ):

        self.method = method
        self.missing_values = missing_values
        self.value = value
        self.forecaster = forecaster
        self.random_state = random_state
        super(Imputer, self).__init__()

        self._check_method()

        if method in self.METHODS_REQUIRE_FIT:
            self.set_tags(fit_is_empty=False)

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        self: a fitted instance of the estimator
        """
        self._check_method()

        if self.method not in self.METHODS_REQUIRE_FIT:
            raise ValueError(
                f"bug, unreachable code: {self.method} passed and _fit was called. "
                "_fit shouhld not be entered since the method does not require _fit."
            )

        if self.method == "mean_fit":
            self._fillconst = X.mean()
        elif self.method == "median_fit":
            self._fillconst = X.median()
        else:
            raise ValueError(f"bug, unreachable code: {self.method} passed.")

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            transformed version of X
        """
        self._check_method()

        # replace missing_values with np.nan
        if self.missing_values:
            X = X.replace(to_replace=self.missing_values, value=np.nan)

        if not _has_missing_values(X):
            return X

        if self.method == "random":
            if isinstance(X, pd.DataFrame):
                Xt = X
                for col in Xt:
                    Xt[col] = Xt[col].apply(
                        lambda i: self._get_random(Xt[col]) if np.isnan(i) else i
                    )
            else:
                Xt = X.apply(lambda i: self._get_random(X) if np.isnan(i) else i)
        elif self.method == "constant":
            Xt = X.fillna(value=self.value)
        elif self.method in ["backfill", "bfill", "pad", "ffill"]:
            Xt = X.fillna(method=self.method)
        elif self.method == "drift":
            forecaster = PolynomialTrendForecaster(degree=1)
            Xt = _impute_with_forecaster(forecaster, X)
        elif self.method == "forecaster":
            forecaster = clone(self.forecaster)
            Xt = _impute_with_forecaster(forecaster, X)
        elif self.method == "mean":
            Xt = X.fillna(value=X.mean())
        elif self.method == "median":
            Xt = X.fillna(value=X.median())
        elif self.method in ["mean_fit", "median_fit"]:
            Xt = X.fillna(value=self._fillconst)
        elif self.method in ["nearest", "linear"]:
            Xt = X.interpolate(method=self.method)
        else:
            raise ValueError(f"`method`: {self.method} not available.")
        # fill first/last elements of series,
        # as some methods (e.g. "linear") cant impute those
        Xt = X.fillna(method="ffill").fillna(method="backfill")
        return Xt

    def _check_method(self):
        if self.method not in self.ALLOWED_METHODS:
            raise ValueError(f"`method`: {self.method} not available.")
        if (
            self.value is not None
            and self.method != "constant"
            or self.method == "constant"
            and self.value is None
        ):
            raise ValueError(
                """Imputing with a value can only be
                used if method="constant" and if parameter "value" is not None"""
            )
        elif (
            self.forecaster is not None
            and self.method != "forecaster"
            or self.method == "forecaster"
            and self.forecaster is None
        ):
            raise ValueError(
                """Imputing with a forecaster can only be used if
                method=\"forecaster\" and if arg forecaster is not None"""
            )
        else:
            pass

    def _get_random(self, Z):
        """Create a random int or float value.

        :param Z: Series
        :type Z: pd.Series
        :return: Random int or float between min and max of Z
        :rtype: int/float
        """
        rng = check_random_state(self.random_state)
        # check if series contains only int or int-like values (e.g. 3.0)
        if (Z.dropna() % 1 == 0).all():
            return rng.randint(Z.min(), Z.max())
        else:
            return rng.uniform(Z.min(), Z.max())

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [{"method": x} for x in cls.ALLOWED_METHODS]

        return params


def _impute_with_forecaster(forecaster, Z):
    """Use a given forecaster for imputation by in-sample predictions.

    Parameters
    ----------
    forecaster: Forecaster
        Forecaster to use for imputation
    Z : pd.Series or pd.DataFrame
        Series to impute.

    Returns
    -------
    zt : pd.Series or pd.DataFrame
        Series with imputed values.
    """
    if isinstance(Z, pd.Series):
        series = [Z]
    elif isinstance(Z, pd.DataFrame):
        series = [Z[column] for column in Z]

    for z in series:
        if _has_missing_values(z):
            # define fh based on index of missing values
            na_index = z.index[z.isna()]
            fh = ForecastingHorizon(values=na_index, is_relative=False)

            # fill NaN before fitting with ffill and backfill (heuristic)
            forecaster.fit(y=z.fillna(method="ffill").fillna(method="backfill"), fh=fh)

            # replace missing values with predicted values
            z[na_index] = forecaster.predict()
    return Z


def _has_missing_values(Z):
    return Z.isnull().to_numpy().any()
