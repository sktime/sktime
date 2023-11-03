#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformations to deseasonalize a timeseries."""

__author__ = ["mloning", "eyalshafran", "aiwalter"]
__all__ = ["Deseasonalizer", "ConditionalDeseasonalizer", "STLTransformer"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.datetime import _get_duration, _get_freq
from sktime.utils.seasonality import autocorrelation_seasonality_test
from sktime.utils.validation.forecasting import check_sp


class Deseasonalizer(BaseTransformer):
    """Remove seasonal components from a time series.

    Applies `statsmodels.tsa.seasonal.seasonal_compose` and removes the `seasonal`
    component in `transform`. Adds seasonal component back again in `inverse_transform`.
    Seasonality removal can be additive or multiplicative.

    `fit` computes :term:`seasonal components <Seasonality>` and
    stores them in `seasonal_` attribute.

    `transform` aligns seasonal components stored in `seasonal_` with
    the time index of the passed :term:`series <Time series>` and then
    subtracts them ("additive" model) from the passed :term:`series <Time series>`
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
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = Deseasonalizer()  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
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
        "python_dependencies": "statsmodels",
    }

    def __init__(self, sp=1, model="additive"):
        self.sp = check_sp(sp)
        allowed_models = ("additive", "multiplicative")
        if model not in allowed_models:
            raise ValueError(
                f"`model` must be one of {allowed_models}, " f"but found: {model}"
            )
        self.model = model
        self._X = None
        self.seasonal_ = None
        super().__init__()

    def _align_seasonal(self, X):
        """Align seasonal components with X's time index."""
        shift = (
            -_get_duration(
                X.index[0],
                self._X.index[0],
                coerce_to_int=True,
                unit=_get_freq(self._X.index),
            )
            % self.sp
        )
        return np.resize(np.roll(self.seasonal_, shift=shift), X.shape[0])

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series
            Data to fit transform to
        y : ignored argument for interface compatibility

        Returns
        -------
        self: a fitted instance of the estimator
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        self._X = X
        sp = self.sp

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
        X_full = X.combine_first(self._X)
        self._X = X_full
        if update_params:
            self._fit(X_full, update_params=update_params)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {}

        params2 = {"sp": 2}

        return [params, params2]


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
    subtracts them ("additive" model) from the passed series
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
    >>> y = load_airline()  # doctest: +SKIP
    >>> transformer = ConditionalDeseasonalizer(sp=12)  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    def __init__(self, seasonality_test=None, sp=1, model="additive"):
        self.seasonality_test = seasonality_test
        self.is_seasonal_ = None
        super().__init__(sp=sp, model=model)

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

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series
            Data to fit transform to
        y : ignored argument for interface compatibility

        Returns
        -------
        self: a fitted instance of the estimator
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        self._X = X
        sp = self.sp

        # set default condition
        if self.seasonality_test is None:
            self.seasonality_test_ = autocorrelation_seasonality_test
        else:
            self.seasonality_test_ = self.seasonality_test

        # check if data meets condition
        self.is_seasonal_ = self._check_condition(X)

        if self.is_seasonal_:
            # if condition is met, apply de-seasonalisation
            self.seasonal_ = seasonal_decompose(
                X,
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

        return self


class STLTransformer(BaseTransformer):
    """Remove seasonal components from a time-series using STL.

    Interfaces ``statsmodels.tsa.seasonal.STL`` as an sktime transformer.

    ``STLTransformer`` can be used to perform deseasonalization or decomposition:

    If ``return_components=False``, it will return the deseasonalized series, i.e.,
    the trend component from ``statsmodels`` ``STL``.

    If ``return_components=True``, it will transform the series into a decomposition
    of component, returning the trend, seasonal, and residual components.

    ``STLTransformer`` performs ``inverse_transform`` by summing any components,
    and can be used for pipelining in a ``TransformedTargetForecaster``.

    Important: for separate forecasts of trend and seasonality, and an
    inverse transform that respects seasonality, ensure
    that ``return_components=True`` is set, otherwise the inverse will just
    return the trend component.

    An alternative for pipeline-style composition is ``STLForecaster``.

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
    return_components : bool, default=False
        if False, will return only the trend component
        if True, will return the transformed series, as well as three components
            as variables in the returned multivariate series (DataFrame cols)
            "transformed" - the transformed series
            "seasonal" - the seasonal component
            "trend" - the trend component
            "resid" - the residuals after de-trending, de-seasonalizing

    Attributes
    ----------
    trend_ : pd.Series
        Trend component of series seen in fit.
    seasonal_ : pd.Series
        Seasonal components of series seen in fit.
    resid_ : pd.Series
        Residuals component of series seen in fit.

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
    >>> X = load_airline()  # doctest: +SKIP
    >>> transformer = STLTransformer(sp=12)  # doctest: +SKIP
    >>> Xt = transformer.fit_transform(X)  # doctest: +SKIP
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "transform-returns-same-time-index": True,
        "univariate-only": True,
        "fit_is_empty": False,
        "python_dependencies": "statsmodels",
        "capability:inverse_transform": True,
        "capability:inverse_transform:exact": False,
        "skip-inverse-transform": False,
    }

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
        return_components=False,
    ):
        self.sp = check_sp(sp)

        # The statsmodels.tsa.seasonal.STL can only deal with sp >= 2
        if sp < 2:
            raise ValueError("sp must be positive integer >= 2")

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
        self.return_components = return_components
        self._X = None
        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series
            Data to fit transform to
        y : ignored argument for interface compatibility

        Returns
        -------
        self: a fitted instance of the estimator
        """
        from statsmodels.tsa.seasonal import STL as _STL

        # remember X for transform
        self._X = X
        sp = self.sp

        self.stl_ = _STL(
            X.values.flatten(),
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

        self.seasonal_ = pd.Series(self.stl_.seasonal, index=X.index)
        self.resid_ = pd.Series(self.stl_.resid, index=X.index)
        self.trend_ = pd.Series(self.stl_.trend, index=X.index)

        return self

    def _transform(self, X, y=None):
        from statsmodels.tsa.seasonal import STL as _STL

        # fit again if indices not seen, but don't store anything
        if not X.index.equals(self._X.index):
            X_full = X.combine_first(self._X)
            new_stl = _STL(
                X_full.values.flatten(),
                period=self.sp,
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

            ret_obj = self._make_return_object(X_full, new_stl)
        else:
            ret_obj = self._make_return_object(X, self.stl_)

        return ret_obj

    def _inverse_transform(self, X, y=None):
        # for inverse transform, we sum up the columns
        # this will be close if return_components=True
        row_sums = X.sum(axis=1)
        row_sums.columns = self.fit_column_names
        return row_sums

    def _make_return_object(self, X, stl):
        # deseasonalize only
        transformed = pd.Series(X.values.flatten() - stl.seasonal, index=X.index)
        # transformed = pd.Series(X.values - stl.seasonal - stl.trend, index=X.index)

        if self.return_components:
            seasonal = pd.Series(stl.seasonal, index=X.index)
            resid = pd.Series(stl.resid, index=X.index)
            trend = pd.Series(stl.trend, index=X.index)

            ret = pd.DataFrame(
                {
                    "transformed": transformed,
                    "seasonal": seasonal,
                    "trend": trend,
                    "resid": resid,
                }
            )
        else:
            ret = pd.DataFrame(transformed, columns=self._X.columns)

        return ret

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
        # test case 1: all default parameters
        params1 = {}

        # test case 2: return all components
        params2 = {"return_components": True}

        # test case 3: seasonality parameter set, from the skipped doctest
        params3 = {"sp": 12}

        return [params1, params2, params3]
