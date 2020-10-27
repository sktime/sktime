# -*- coding: utf-8 -*-
__all__ = ["ExponentialSmoothing"]
__author__ = ["Markus Löning", "@big-o"]

from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ExponentialSmoothing

from sktime.forecasting.base._adapters import _StatsModelsAdapter


class ExponentialSmoothing(_StatsModelsAdapter):
    """
    Holt-Winters exponential smoothing forecaster. Default settings use
    simple exponential smoothing
    without trend and seasonality components.

    Parameters
    ----------
    trend : str{"add", "mul", "additive", "multiplicative", None}, optional
    (default=None)
        Type of trend component.
    damped : bool, optional (default=None)
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, optional
    (default=None)
        Type of seasonal component.
    sp : int, optional (default=None)
        The number of seasons to consider for the holt winters.
    smoothing_level : float, optional
        The alpha value of the simple exponential smoothing, if the value
        is set then this value will be used as the value.
    smoothing_slope : float, optional
        The beta value of the Holt's trend method, if the value is
        set then this value will be used as the value.
    smoothing_seasonal : float, optional
        The gamma value of the holt winters seasonal method, if the value
        is set then this value will be used as the value.
    damping_slope : float, optional
        The phi value of the damped method, if the value is
        set then this value will be used as the value.
    optimized : bool, optional
        Estimate model parameters by maximizing the log-likelihood
    use_boxcox : {True, False, 'log', float}, optional
        Should the Box-Cox transform be applied to the data first? If 'log'
        then apply the log. If float then use lambda equal to float.
    remove_bias : bool, optional
        Remove bias from forecast values and fitted values by enforcing
        that the average residual is equal to zero.
    use_basinhopping : bool, optional
        Using Basin Hopping optimizer to find optimal values

    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    _fitted_param_names = (
        "initial_level",
        "initial_slope",
        "initial_seasons",
        "smoothing_level",
        "smoothing_slope",
        "smoothing_seasonal",
        "damping_slope",
    )

    def __init__(
        self,
        trend=None,
        damped=False,
        seasonal=None,
        sp=None,
        smoothing_level=None,
        smoothing_slope=None,
        smoothing_seasonal=None,
        damping_slope=None,
        optimized=True,
        use_boxcox=False,
        remove_bias=False,
        use_basinhopping=False,
    ):
        # Model params
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.sp = sp

        # Fit params
        self.smoothing_level = smoothing_level
        self.optimized = optimized
        self.smoothing_slope = smoothing_slope
        self.smoothing_seasonal = smoothing_seasonal
        self.damping_slope = damping_slope
        self.use_boxcox = use_boxcox
        self.remove_bias = remove_bias
        self.use_basinhopping = use_basinhopping

        super(ExponentialSmoothing, self).__init__()

    def _fit_forecaster(self, y, X=None):
        self._forecaster = _ExponentialSmoothing(
            y,
            trend=self.trend,
            damped=self.damped,
            seasonal=self.seasonal,
            seasonal_periods=self.sp,
        )

        self._fitted_forecaster = self._forecaster.fit(
            smoothing_level=self.smoothing_level,
            optimized=self.optimized,
            smoothing_slope=self.smoothing_slope,
            smoothing_seasonal=self.smoothing_seasonal,
            damping_slope=self.damping_slope,
            use_boxcox=self.use_boxcox,
            remove_bias=self.remove_bias,
            use_basinhopping=self.use_basinhopping,
        )
