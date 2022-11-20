# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Holt-Winters exponential smoothing."""

__all__ = ["ExponentialSmoothing"]
__author__ = ["mloning", "big-o"]

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class ExponentialSmoothing(_StatsModelsAdapter):
    """Holt-Winters exponential smoothing forecaster.

    Direct interface for `statsmodels.tsa.holtwinters`.

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
    smoothing_level : float, optional
        The alpha value of the simple exponential smoothing, if the value
        is set then this value will be used as the value.
    smoothing_trend :  float, optional
        The beta value of the Holt's trend method, if the value is
        set then this value will be used as the value.
    smoothing_seasonal : float, optional
        The gamma value of the holt winters seasonal method, if the value
        is set then this value will be used as the value.
    damping_trend : float, optional
        The phi value of the damped method, if the value is
        set then this value will be used as the value.
    optimized : bool, optional
        Estimate model parameters by maximizing the log-likelihood.
    remove_bias : bool, optional
        Remove bias from forecast values and fitted values by enforcing
        that the average residual is equal to zero.
    start_params : array_like, optional
        Starting values to used when optimizing the fit.  If not provided,
        starting values are determined using a combination of grid search
        and reasonable values based on the initial values of the data. See
        the notes for the structure of the model parameters.
    method : str, default "L-BFGS-B"
        The minimizer used. Valid options are "L-BFGS-B" , "TNC",
        "SLSQP" (default), "Powell", "trust-constr", "basinhopping" (also
        "bh") and "least_squares" (also "ls"). basinhopping tries multiple
        starting values in an attempt to find a global minimizer in
        non-convex problems, and so is slower than the others.
    minimize_kwargs : dict[str, Any]
        A dictionary of keyword arguments passed to SciPy's minimize
        function if method is one of "L-BFGS-B", "TNC",
        "SLSQP", "Powell", or "trust-constr", or SciPy's basinhopping
        or least_squares functions. The valid keywords are optimizer
        specific. Consult SciPy's documentation for the full set of
        options.
    use_brute : bool, optional
        Search for good starting values using a brute force (grid)
        optimizer. If False, a naive set of starting values is used.
    random_state : int, RandomState instance or None, optional ,
        default=None – If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by np.random.

    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    >>> y = load_airline()
    >>> forecaster = ExponentialSmoothing(
    ...     trend='add', seasonal='multiplicative', sp=12
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    ExponentialSmoothing(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
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
        smoothing_level=None,
        smoothing_trend=None,
        smoothing_seasonal=None,
        damping_trend=None,
        optimized=True,
        remove_bias=False,
        start_params=None,
        method=None,
        minimize_kwargs=None,
        use_brute=True,
        random_state=None,
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
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.damping_trend = damping_trend
        self.optimized = optimized
        self.remove_bias = remove_bias
        self.start_params = start_params
        self.method = method
        self.minimize_kwargs = minimize_kwargs
        self.use_brute = use_brute

        super().__init__(random_state=random_state)

    def _fit_forecaster(self, y, X=None):
        from statsmodels.tsa.holtwinters import (
            ExponentialSmoothing as _ExponentialSmoothing,
        )

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

        self._fitted_forecaster = self._forecaster.fit(
            smoothing_level=self.smoothing_level,
            smoothing_trend=self.smoothing_trend,
            smoothing_seasonal=self.smoothing_seasonal,
            damping_trend=self.damping_trend,
            optimized=self.optimized,
            remove_bias=self.remove_bias,
            start_params=self.start_params,
            method=self.method,
            minimize_kwargs=self.minimize_kwargs,
            use_brute=self.use_brute,
        )
