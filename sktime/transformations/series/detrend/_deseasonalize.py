#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "Deseasonalizer",
    "Deseasonalizer",
    "ConditionalDeseasonalizer",
    "ConditionalDeseasonalizer",
]

import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.datetime import _get_duration
from sktime.utils.datetime import _get_freq
from sktime.utils.seasonality import autocorrelation_seasonality_test
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.validation.series import check_series


class Deseasonalizer(_SeriesToSeriesTransformer):
    """A transformer that removes seasonal components from time
    series.

    Parameters
    ----------
    sp : int, optional (default=1)
        Seasonal periodicity
    model : str {"additive", "multiplicative"}, optional (default="additive")
        Model to use for estimating seasonal component

    Example
    ----------
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Deseasonalizer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {"transform-returns-same-time-index": True, "univariate-only": True}

    def __init__(self, sp=1, model="additive"):
        self.sp = check_sp(sp)
        allowed_models = ("additive", "multiplicative")
        if model not in allowed_models:
            raise ValueError(
                f"`model` must be one of {allowed_models}, " f"but found: {model}"
            )
        self.model = model
        self._y_index = None
        self.seasonal_ = None
        super(Deseasonalizer, self).__init__()

    def _set_y_index(self, y):
        self._y_index = y.index

    def _align_seasonal(self, y):
        """Align seasonal components with y's time index"""
        shift = (
            -_get_duration(
                y.index[0],
                self._y_index[0],
                coerce_to_int=True,
                unit=_get_freq(self._y_index),
            )
            % self.sp
        )
        return np.resize(np.roll(self.seasonal_, shift=shift), y.shape[0])

    def fit(self, Z, X=None):
        """Fit to data.

        Parameters
        ----------
        Z : pd.Series
        X : pd.DataFrame

        Returns
        -------
        self : an instance of self
        """
        self._is_fitted = False
        z = check_series(Z, enforce_univariate=True)
        self._set_y_index(z)
        sp = check_sp(self.sp)

        # apply seasonal decomposition
        self.seasonal_ = seasonal_decompose(
            z,
            model=self.model,
            period=sp,
            filt=None,
            two_sided=True,
            extrapolate_trend=0,
        ).seasonal.iloc[:sp]

        self._is_fitted = True
        return self

    def _transform(self, y, seasonal):
        if self.model == "additive":
            return y - seasonal
        else:
            return y / seasonal

    def _inverse_transform(self, y, seasonal):
        if self.model == "additive":
            return y + seasonal
        else:
            return y * seasonal

    def transform(self, Z, X=None):
        """Transform data.
        Returns a transformed version of y.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame

        Returns
        -------
        yt : pd.Series
            Transformed time series.
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        seasonal = self._align_seasonal(z)
        return self._transform(z, seasonal)

    def inverse_transform(self, Z, X=None):
        """Inverse transform data.
        Returns a transformed version of y.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame

        Returns
        -------
        yt : pd.Series
            Transformed time series.
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        seasonal = self._align_seasonal(z)
        return self._inverse_transform(z, seasonal)

    def update(self, Z, X=None, update_params=False):
        """Update fitted parameters

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=False)

        Returns
        -------
        self : an instance of self
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        self._set_y_index(z)
        return self


class ConditionalDeseasonalizer(Deseasonalizer):
    """A transformer that removes seasonal components from time
    series, conditional on seasonality test.

    Parameters
    ----------
    seasonality_test : callable, optional (default=None)
        Callable that tests for seasonality and returns True when data is
        seasonal and False otherwise. If None,
        90% autocorrelation seasonality test is used.
    sp : int, optional (default=1)
        Seasonal periodicity
    model : str {"additive", "multiplicative"}, optional (default="additive")
        Model to use for estimating seasonal component
    """

    def __init__(self, seasonality_test=None, sp=1, model="additive"):
        self.seasonality_test = seasonality_test
        self.is_seasonal_ = None
        super(ConditionalDeseasonalizer, self).__init__(sp=sp, model=model)

    def _check_condition(self, y):
        """Check if y meets condition"""

        if not callable(self.seasonality_test_):
            raise ValueError(
                f"`func` must be a function/callable, but found: "
                f"{type(self.seasonality_test_)}"
            )

        is_seasonal = self.seasonality_test_(y, sp=self.sp)
        if not isinstance(is_seasonal, (bool, np.bool_)):
            raise ValueError(
                f"Return type of `func` must be boolean, "
                f"but found: {type(is_seasonal)}"
            )
        return is_seasonal

    def fit(self, Z, X=None):
        """Fit to data.

        Parameters
        ----------
        y_train : pd.Series

        Returns
        -------
        self : an instance of self
        """

        z = check_series(Z, enforce_univariate=True)
        self._set_y_index(z)
        sp = check_sp(self.sp)

        # set default condition
        if self.seasonality_test is None:
            self.seasonality_test_ = autocorrelation_seasonality_test
        else:
            self.seasonality_test_ = self.seasonality_test

        # check if data meets condition
        self.is_seasonal_ = self._check_condition(z)

        if self.is_seasonal_:
            # if condition is met, apply de-seasonalisation
            self.seasonal_ = seasonal_decompose(
                z,
                model=self.model,
                period=sp,
                filt=None,
                two_sided=True,
                extrapolate_trend=0,
            ).seasonal.iloc[:sp]
        else:
            # otherwise, set idempotent seasonal components
            self.seasonal_ = (
                np.zeros(self.sp) if self.model == "additive" else np.ones(self.sp)
            )

        self._is_fitted = True
        return self
