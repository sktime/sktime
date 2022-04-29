#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformations to deseasonalize a timeseries."""

__author__ = ["mloning", "eyalshafran", "aiwalter"]
__all__ = ["Deseasonalizer", "ConditionalDeseasonalizer", "STLTransformer"]

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL as _STL
from statsmodels.tsa.seasonal import seasonal_decompose

from sktime.transformations.base import BaseTransformer, _SeriesToSeriesTransformer
from sktime.utils.datetime import _get_duration, _get_freq
from sktime.utils.seasonality import autocorrelation_seasonality_test
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.validation.series import check_series


class Deseasonalizer(BaseTransformer):
    """Remove seasonal components from a time series.

    Fit computes :term:`seasonal components <Seasonality>` and
    stores them in `seasonal_`.

    Transform aligns seasonal components stored in `_seasonal` with
    the time index of the passed :term:`series <Time series>` and then
    substracts them ("additive" model) from the passed :term:`series <Time series>`
    or divides the passed series by them ("multiplicative" model).

    Parameters
    ----------
    sp : int, default=1
        Seasonal periodicity.
    model : {"additive", "multiplicative"}, default="additive"
        Model to use for estimating seasonal component.

    Attributes
    ----------
    seasonal_ : array of length sp
        Seasonal components computed in seasonal decomposition.

    See Also
    --------
    ConditionalDeseasonalizer

    Notes
    -----
    For further explanation on seasonal components and additive vs.
    multiplicative models see
    `Forecasting: Principles and Practice <https://otexts.com/fpp3/components.html>`_.
    Seasonal decomposition is computed using `statsmodels
    <https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html>`_.

    Examples
    --------
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = Deseasonalizer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.Series",
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,
        "capability:inverse_transform": True,
        "transform-returns-same-time-index": True,
        "univariate-only": True,
    }

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
        """Align seasonal components with y's time index."""
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

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        self._set_y_index(X)
        sp = check_sp(self.sp)

        # apply seasonal decomposition
        self.seasonal_ = seasonal_decompose(
            X,
            model=self.model,
            period=sp,
            filt=None,
            two_sided=True,
            extrapolate_trend=0,
        ).seasonal.iloc[:sp]
        return self

    def _private_transform(self, y, seasonal):
        if self.model == "additive":
            return y - seasonal
        else:
            return y / seasonal

    def _private_inverse_transform(self, y, seasonal):
        if self.model == "additive":
            return y + seasonal
        else:
            return y * seasonal

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series
            transformed version of X, detrended series
        """
        seasonal = self._align_seasonal(X)
        Xt = self._private_transform(X, seasonal)
        return Xt

    def _inverse_transform(self, X, y=None):
        """Logic used by `inverse_transform` to reverse transformation on `X`.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be inverse transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            inverse transformed version of X
        """
        seasonal = self._align_seasonal(X)
        Xt = self._private_inverse_transform(X, seasonal)
        return Xt

    def _update(self, X, y=None, update_params=False):
        """Update transformer with X and y.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X : pd.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        self._set_y_index(X)
        return self


class ConditionalDeseasonalizer(Deseasonalizer):
    """Remove seasonal components from time series, conditional on seasonality test.

    Fit tests for :term:`seasonality <Seasonality>` and if the passed time series
    has a seasonal component it applies seasonal decomposition provided by `statsmodels
    <https://www.statsmodels.org>`
    to compute the seasonal component.
    If the test is negative `_seasonal` is set
    to all ones (if `model` is "multiplicative")
    or to all zeros (if `model` is "additive").

    Transform aligns seasonal components stored in `seasonal_` with
    the time index of the passed series and then
    substracts them ("additive" model) from the passed series
    or divides the passed series by them ("multiplicative" model).


    Parameters
    ----------
    seasonality_test : callable or None, default=None
        Callable that tests for seasonality and returns True when data is
        seasonal and False otherwise. If None,
        90% autocorrelation seasonality test is used.
    sp : int, default=1
        Seasonal periodicity.
    model : {"additive", "multiplicative"}, default="additive"
        Model to use for estimating seasonal component.

    Attributes
    ----------
    seasonal_ : array of length sp
        Seasonal components.
    is_seasonal_ : bool
        Return value of `seasonality_test`. True when data is
        seasonal and False otherwise.

    See Also
    --------
    Deseasonalizer

    Notes
    -----
    For further explanation on seasonal components and additive vs.
    multiplicative models see
    `Forecasting: Principles and Practice <https://otexts.com/fpp3/components.html>`_.
    Seasonal decomposition is computed using `statsmodels
    <https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html>`_.

    Examples
    --------
    >>> from sktime.transformations.series.detrend import ConditionalDeseasonalizer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = ConditionalDeseasonalizer(sp=12)
    >>> y_hat = transformer.fit_transform(y)
    """

    def __init__(self, seasonality_test=None, sp=1, model="additive"):
        self.seasonality_test = seasonality_test
        self.is_seasonal_ = None
        super(ConditionalDeseasonalizer, self).__init__(sp=sp, model=model)

    def _check_condition(self, y):
        """Check if y meets condition."""
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


class STLTransformer(_SeriesToSeriesTransformer):
    """Remove seasonal components from a time-series using STL.

    The STLTransformer is a descriptive transformer to remove seasonality
    from a series and is based on statsmodels.STL. It returns deseasonalized
    data. All three components trend, season and residuals can be accessed
    via attributes trend_, season_ and resid_. STLTransformer can not transform
    or inverse_transform on data that was not given in fit() before.
    This means that for pipelining, the Deseasonalizer or Detrender must be
    used instead of STLTransformer.

    Parameters
    ----------
    sp : int, default=1
        Seasonal periodicity.
    seasonal : int, default=7
        Length of the seasonal smoother. Must be an odd integer, and should
        normally be >= 7 (default).
    trend : {int, default=None}
        Length of the trend smoother. Must be an odd integer. If not provided
        uses the smallest odd integer greater than
        1.5 * period / (1 - 1.5 / seasonal), following the suggestion in
        the original implementation.
    low_pass : {int, default=None}
        Length of the low-pass filter. Must be an odd integer >=3. If not
        provided, uses the smallest odd integer > period.
    seasonal_deg : int, default=1
        Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend).
    trend_deg : int, default=1
        Degree of trend LOESS. 0 (constant) or 1 (constant and trend).
    low_pass_deg : int, default=1
        Degree of low pass LOESS. 0 (constant) or 1 (constant and trend).
    robust : bool, default False
        Flag indicating whether to use a weighted version that is robust to
        some forms of outliers.
    seasonal_jump : int, default=1
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every seasonal_jump points and linear
        interpolation is between fitted points. Higher values reduce
        estimation time.
    trend_jump : int, default=1
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every trend_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation
        time.
    low_pass_jump : int, default=1
        Positive integer determining the linear interpolation step. If larger
        than 1, the LOESS is used every low_pass_jump points and values between
        the two are linearly interpolated. Higher values reduce estimation
        time.

    Attributes
    ----------
    trend_ : pd.Series
        Trend component.
    seasonal_ : pd.Series
        Seasonal components.
    resid_ : pd.Series
        Residuals component.

    See Also
    --------
    Detrender
    Deseasonalizer
    STLForecaster

    References
    ----------
    .. [1] https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.STL.html

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.detrend import STLTransformer
    >>> y = load_airline()
    >>> transformer = STLTransformer(sp=12)
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {"transform-returns-same-time-index": True, "univariate-only": True}

    def __init__(
        self,
        sp=2,
        seasonal=7,
        trend=None,
        low_pass=None,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        robust=False,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
    ):
        self.sp = check_sp(sp)
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.low_pass_jump = low_pass_jump
        self._Z_index = None
        super(STLTransformer, self).__init__()

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
        self._Z_index = Z.index
        sp = check_sp(self.sp)

        # The statsmodels.tsa.seasonal.STL can only deal with sp >= 2
        if sp < 2:
            raise ValueError("sp must be positive integer >= 2")
        self._stl = _STL(
            z.values,
            period=sp,
            seasonal=self.seasonal,
            trend=self.trend,
            low_pass=self.low_pass,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            low_pass_deg=self.low_pass_deg,
            robust=self.robust,
            seasonal_jump=self.seasonal_jump,
            trend_jump=self.trend_jump,
            low_pass_jump=self.low_pass_jump,
        ).fit()

        self.seasonal_ = pd.Series(self._stl.seasonal, index=Z.index)
        self.resid_ = pd.Series(self._stl.resid, index=Z.index)
        self.trend_ = pd.Series(self._stl.trend, index=Z.index)

        self._is_fitted = True
        return self

    def _transform(self, y):
        if not self._Z_index.equals(y.index):
            raise NotImplementedError(
                """
                STLTransformer is only a descriptive trasnformer and
                can only transform data that was given in fit().
                Please use Deseasonalizer or Detrender."""
            )
        return y - self.seasonal_

    def _inverse_transform(self, y):
        if not self._Z_index.equals(y.index):
            raise NotImplementedError(
                """
                STLTransformer is only a descriptive trasnformer and
                can only inverse_transform data that was given in fit().
                Please use Deseasonalizer or Detrender."""
            )
        return y + self.seasonal_

    def transform(self, Z, X=None):
        """Transform data.

        Returns a transformed version yt of y. The seasonal component is removed from y.
        The trend and residual components can be accessed via
        the attributes trend_ and resid_ for the fitted data.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame

        Returns
        -------
        yt : pd.Series
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        return self._transform(z)

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
        return self._inverse_transform(z)
