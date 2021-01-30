#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Martin Walter"]
__all__ = ["Imputer"]

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series
from sktime.forecasting.trend import PolynomialTrendForecaster


import numpy as np
import random


class Imputer(_SeriesToSeriesTransformer):
    """Missing value imputation

    Parameters
    ----------
    method : {"backfill", "bfill", "pad", "ffill", "random",
        "value", "drift", "mean", "median"}
        Method to of filling values.
            * backfill/bfill : adapted from pd.Series.fillna()
            * pad/ffill : adapted from pd.Series.fillna()
            * random : random values between pd.Series.min() and .max()
            * value : same value for all NaN
            * drift : drift (trend) values by sktime.PolynomialTrendForecaster
            * mean : pd.Series.mean()
            * median : pd.Series.median()
    random_state : int/float/str, optional
        Value to set random.seed()
    value : int/float, optional
        Value to fill NaN, by default None
    """

    def __init__(self, method, random_state=None, value=None):

        self.method = method
        self.random_state = random_state
        self.value = value
        super(Imputer, self).__init__()

    def transform(self, Z, X=None):
        self.check_is_fitted()
        self._check_method()
        z = check_series(Z, enforce_univariate=True)

        if self.method == "random":
            z = z.apply(lambda x: self._get_random(z) if np.isnan(x) else x)
        elif self.method == "value":
            z = z.fillna(value=self.value)
        elif self.method in ["backfill", "bfill", "pad", "ffill"]:
            z = z.fillna(method=self.method)
        elif self.method == "drift":
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
        else:
            raise ValueError(f"method {self.method} not available")
        return z

    def _check_method(self):
        if (
            self.value is not None
            and self.method != "value"
            or self.method == "value"
            and self.value is None
        ):
            raise ValueError(
                """Imputing with a value can only be
                used if method=\"value\" and if value is not None"""
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
        random.seed(self.random_state)
        # check if series contains only int or int-like values (e.g. 3.0)
        if (z.dropna() % 1 == 0).all():
            return random.randint(z.min(), z.max())
        else:
            return random.uniform(z.min(), z.max())
