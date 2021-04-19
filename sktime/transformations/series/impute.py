#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Martin Walter"]
__all__ = ["Imputer"]

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series
from sktime.forecasting.trend import PolynomialTrendForecaster
from sklearn.utils import check_random_state


import numpy as np
import pandas as pd


class Imputer(_SeriesToSeriesTransformer):
    """Missing value imputation

    Parameters
    ----------
    method : Method to fill values.
        * "drift" : drift/trend values by sktime.PolynomialTrendForecaster()
        * "linear" : linear interpolation, by pd.Series.interpolate()
        * "nearest" : use nearest value, by pd.Series.interpolate()
        * "constant" : same constant value (given in arg value) for all NaN
        * "mean" : pd.Series.mean()
        * "median" : pd.Series.median()
        * "backfill"/"bfill" : adapted from pd.Series.fillna()
        * "pad"/"ffill" : adapted from pd.Series.fillna()
        * "random" : random values between pd.Series.min() and .max()
        * "forecaster" : use an sktime Forecaster, given in arg forecaster
    missing_values : int/float/str, optional
        The placeholder for the missing values. All occurrences of
        missing_values will be imputed. Default, None (np.nan)
    value : int/float, optional
        Value to fill NaN, by default None
    forecaster : Any Forecaster based on sktime.BaseForecaster, optinal
        Use a given Forecaster to impute by insample predictions. Before
        fitting, missing data is imputed with method="ffill"/"bfill"
        as heuristic.
    random_state : int/float/str, optional
        Value to set random.seed() if method="random", default None
    """

    _tags = {
        "univariate-only": True,
        "fit-in-transform": True,
        "handles-missing-data": True,
    }

    def __init__(
        self,
        method,
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

    def transform(self, Z, X=None):
        """Transform data.
        Returns a transformed version of Z.

        Parameters
        ----------
        Z : pd.Series

        Returns
        -------
        Z : pd.Series
            Transformed time series.
        """
        self.check_is_fitted()
        self._check_method()
        Z = check_series(Z)
        # multivariate
        if isinstance(Z, pd.DataFrame):
            for col in Z:
                Z[col] = self._transform_series(Z[col])
        # univariate
        else:
            Z = self._transform_series(Z)
        return Z

    def _transform_series(self, Z):
        # replace missing_values with np.nan
        if self.missing_values:
            Z = Z.replace(to_replace=self.missing_values, value=np.nan)

        if self.method == "random":
            Z = Z.apply(lambda x: self._get_random(Z) if np.isnan(x) else x)
        elif self.method == "constant":
            Z = Z.fillna(value=self.value)
        elif self.method in ["backfill", "bfill", "pad", "ffill"]:
            Z = Z.fillna(method=self.method)
        elif self.method in ["drift", "forecaster"]:
            if self.method == "forecaster":
                forecaster = self.forecaster
            else:
                forecaster = PolynomialTrendForecaster(degree=1)
            # in-sample forecasting horizon
            fh_ins = -np.arange(len(Z))
            # fill NaN before fitting with ffill and backfill (heuristic)
            Z_pred = forecaster.fit(
                Z.fillna(method="ffill").fillna(method="backfill")
            ).predict(fh=fh_ins)
            # fill with trend values
            Z = Z.fillna(value=Z_pred)
        elif self.method == "mean":
            Z = Z.fillna(value=Z.mean())
        elif self.method == "median":
            Z = Z.fillna(value=Z.median())
        elif self.method in ["nearest", "linear"]:
            Z = Z.interpolate(method=self.method)
        else:
            raise ValueError(f"method {self.method} not available")
        # fill first/last elements of series,
        # as some methods (e.g. "linear") cant impute those
        Z = Z.fillna(method="ffill").fillna(method="backfill")
        return Z

    def _check_method(self):
        if (
            self.value is not None
            and self.method != "constant"
            or self.method == "constant"
            and self.value is None
        ):
            raise ValueError(
                """Imputing with a value can only be
                used if method=\"value\" and if arg value is not None"""
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
        """Create a random int or float value

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
