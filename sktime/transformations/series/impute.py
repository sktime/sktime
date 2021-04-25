#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Martin Walter"]
__all__ = ["Imputer"]

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series
from sktime.forecasting.trend import PolynomialTrendForecaster
from sklearn.utils import check_random_state


import numpy as np


class Imputer(_SeriesToSeriesTransformer):
    """Missing value imputation

    Parameters
    ----------
    method : str, optional (default="drift")
        Method to fill the missing values values.
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

    Example
    ----------
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Imputer(method="drift")
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "univariate-only": True,
        "fit-in-transform": True,
        "handles-missing-data": True,
        "skip-inverse-transform": True,
    }

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

    def transform(self, Z, X=None):
        """Transform data.
        Returns a transformed version of Z.

        Parameters
        ----------
        Z : pd.Series

        Returns
        -------
        z : pd.Series
            Transformed time series.
        """
        self.check_is_fitted()
        self._check_method()
        z = check_series(Z, enforce_univariate=True)

        # replace missing_values with np.nan
        if self.missing_values:
            z = z.replace(to_replace=self.missing_values, value=np.nan)

        if self.method == "random":
            z = z.apply(lambda x: self._get_random(z) if np.isnan(x) else x)
        elif self.method == "constant":
            z = z.fillna(value=self.value)
        elif self.method in ["backfill", "bfill", "pad", "ffill"]:
            z = z.fillna(method=self.method)
        elif self.method in ["drift", "forecaster"]:
            if self.method == "forecaster":
                forecaster = self.forecaster
            else:
                forecaster = PolynomialTrendForecaster(degree=1)
            # in-sample forecasting horizon
            fh_ins = -np.arange(len(z))
            # fill NaN before fitting with ffill and backfill (heuristic)
            z_pred = forecaster.fit(
                z.fillna(method="ffill").fillna(method="backfill")
            ).predict(fh=fh_ins)
            # fill with trend values
            z = z.fillna(value=z_pred)
        elif self.method == "mean":
            z = z.fillna(value=z.mean())
        elif self.method == "median":
            z = z.fillna(value=z.median())
        elif self.method in ["nearest", "linear"]:
            z = z.interpolate(method=self.method)
        else:
            raise ValueError(f"method {self.method} not available")
        return z

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

    def _get_random(self, z):
        """Create a random int or float value

        :param z: Series
        :type z: pd.Series
        :return: Random int or float between min and max of z
        :rtype: int/float
        """
        rng = check_random_state(self.random_state)
        # check if series contains only int or int-like values (e.g. 3.0)
        if (z.dropna() % 1 == 0).all():
            return rng.randint(z.min(), z.max())
        else:
            return rng.uniform(z.min(), z.max())
