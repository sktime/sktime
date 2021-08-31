# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Holt-Winters exponential smoothing."""

__all__ = ["ExponentialSmoothing"]
__author__ = ["mloning", "big-o"]

from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ExponentialSmoothing

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class ExponentialSmoothing(_StatsModelsAdapter):
    """Holt-Winters exponential smoothing forecaster.

    Default settings use simple exponential smoothing without trend and
    seasonality components.

    Parameters
    ----------
    trend : {"add", "mul", "additive", "multiplicative", None}, default=None
        Type of trend component.
    damped_trend : bool, default=False
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, default=None
        Type of seasonal component.Takes one of
    sp : int or None, default=None
        The number of seasonal periods to consider.
    initial_level : float or None, default=None
        The alpha value of the simple exponential smoothing, if the value
        is set then this value will be used as the value.
    initial_trend : float or None, default=None
        The beta value of the Holt's trend method, if the value is
        set then this value will be used as the value.
    initial_seasonal : float or None, default=None
        The gamma value of the holt winters seasonal method, if the value
        is set then this value will be used as the value.
    use_boxcox : {True, False, 'log', float}, default=None
        Should the Box-Cox transform be applied to the data first?
        If 'log' then apply the log. If float then use lambda equal to float.
    initialization_method:{'estimated','heuristic','legacy-heuristic','known',None},
        default='estimated'
        Method for initialize the recursions.
        If 'known' initialization is used, then `initial_level` must be
        passed, as well as `initial_trend` and `initial_seasonal` if
        applicable.
        'heuristic' uses a heuristic based on the data to estimate initial
        level, trend, and seasonal state. 'estimated' uses the same heuristic
        as initial guesses, but then estimates the initial states as part of
        the fitting process.

    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    >>> y = load_airline()
    >>> forecaster = ExponentialSmoothing(trend='add', seasonal='multiplicative', sp=12)
    >>> forecaster.fit(y)
    ExponentialSmoothing(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _fitted_param_names = (
        "initial_level",
        "initial_slope",
        "initial_seasons",
        "aic",
        "bic",
        "aicc",
    )

    def __init__(
        self,
        trend=None,
        damped_trend=False,
        seasonal=None,
        sp=None,
        initial_level=None,
        initial_trend=None,
        initial_seasonal=None,
        use_boxcox=None,
        initialization_method="estimated",
    ):
        # Model params
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.sp = sp
        self.use_boxcox = use_boxcox
        self.initial_level = initial_level
        self.initial_trend = initial_trend
        self.initial_seasonal = initial_seasonal
        self.initialization_method = initialization_method

        super(ExponentialSmoothing, self).__init__()

    def _fit_forecaster(self, y, X=None):
        self._forecaster = _ExponentialSmoothing(
            y,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.sp,
            use_boxcox=self.use_boxcox,
            initial_level=self.initial_level,
            initial_trend=self.initial_trend,
            initial_seasonal=self.initial_seasonal,
            initialization_method=self.initialization_method,
        )

        self._fitted_forecaster = self._forecaster.fit()
