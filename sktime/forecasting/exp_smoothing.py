# -*- coding: utf-8 -*-
__all__ = ["ExponentialSmoothing"]
__author__ = ["Markus Löning", "@big-o"]

from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ExponentialSmoothing

from sktime.forecasting.base.adapters import _StatsModelsAdapter


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
    damped_trend : bool, optional (default=None)
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, optional
    (default=None)
        Type of seasonal component.
    sp : int, optional (default=None)
        The number of seasonal periods to consider.
    initial_level : float, optional
        The alpha value of the simple exponential smoothing, if the value
        is set then this value will be used as the value.
    initial_trend : float, optional
        The beta value of the Holt's trend method, if the value is
        set then this value will be used as the value.
    initial_seasonal : float, optional
        The gamma value of the holt winters seasonal method, if the value
        is set then this value will be used as the value.
    use_boxcox : {True, False, 'log', float}, optional
        Should the Box-Cox transform be applied to the data first? If 'log'
        then apply the log. If float then use lambda equal to float.

    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.

    Example
    ----------
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
