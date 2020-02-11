__all__ = ["ExpSmoothingForecaster"]
__author__ = ["Markus Löning", "@big-o"]

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base import OptionalForecastingHorizonMixin, DEFAULT_ALPHA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ExpSmoothingForecaster(OptionalForecastingHorizonMixin, BaseForecaster):
    """
    Holt-Winters exponential smoothing forecaster. Default settings use simple exponential smoothing
    without trend and seasonality components.

    Parameters
    ----------
    trend : str{"add", "mul", "additive", "multiplicative", None}, optional (default=None)
        Type of trend component.
    damped : bool, optional (default=None)
        Should the trend component be damped.
    seasonal : {"add", "mul", "additive", "multiplicative", None}, optional (default=None)
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

        super(ExpSmoothingForecaster, self).__init__()

    def fit(self, y_train, fh=None, X_train=None):
        """
        Fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : array-like, optional (default=[1])
            The forecasters horizon with the steps ahead to to predict.
        X_train : None
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """

        # update observation horizon
        self._set_oh(y_train)
        self._set_fh(fh)

        self._fit_estimator(y_train)

        self._is_fitted = True
        return self

    def _fit_estimator(self, y):
        # Fit forecaster.
        self._estimator = ExponentialSmoothing(
            y,
            trend=self.trend,
            damped=self.damped,
            seasonal=self.seasonal,
            seasonal_periods=self.sp,
        )

        self._fitted_estimator = self._estimator.fit(
            smoothing_level=self.smoothing_level,
            optimized=self.optimized,
            smoothing_slope=self.smoothing_slope,
            smoothing_seasonal=self.smoothing_seasonal,
            damping_slope=self.damping_slope,
            use_boxcox=self.use_boxcox,
            remove_bias=self.remove_bias,
            use_basinhopping=self.use_basinhopping,
        )

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """
        Make forecasts.

        Parameters
        ----------
        fh : int or array-like, optional (default=1)
            The forecasters horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])
        X : None
            Exogenous variables are ignored.
        return_pred_int : bool, optional (default=False)
        alpha : int or list, optional (default=0.95)

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """
        if return_pred_int:
            raise NotImplementedError()

        # Input checks.
        self._check_is_fitted()

        # Set forecast horizon.
        self._set_fh(fh)
        fh_abs = self._get_absolute_fh()

        # Predict fitted model with start and end points relative to start of train series
        start = fh_abs[0]
        end = fh_abs[-1]
        y_pred = self._fitted_estimator.predict(start=start, end=end)

        # Forecast all periods from start to end of pred horizon,
        # but only return given time points in pred horizon
        return y_pred.loc[fh_abs]

    def update(self, y_new, X_new=None, update_params=True):
        # input checks
        self._check_is_fitted()

        # update observation horizon
        self._set_oh(y_new)

        if update_params:
            self._fit_estimator(y_new)

        return self
