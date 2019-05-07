from .base import BaseSingleSeriesForecaster
from .base import BaseForecaster
from .base import BaseUpdateableForecaster

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
# SARIMAX is better maintained and offers the same functionality as standard ARIMA class
# https://github.com/statsmodels/statsmodels/issues/3884


__all__ = ["ARIMAForecaster", "ExpSmoothingForecaster", "DummyForecaster", "EnsembleForecaster"]
__author__ = ['Markus Löning']


class ARIMAForecaster(BaseUpdateableForecaster):
    """
    ARIMA (Auto-regressive Integrated Moving-average) forecaster.

    Parameters
    ----------
    order : tuple, optional (default=(1, 0, 0))
        The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters. d must be an
        integer indicating the integration order of the process, while p and q may either be an integers indicating the
        AR and MA orders (so that all lags up to those orders are included) or else iterables giving specific AR and /
        or MA lags to include. Default is an AR(1) model: (1,0,0).
    seasonal_order : tuple, optional (default=(0, 0, 0, 0))
        The (P,D,Q,s) order of the seasonal component of the model for the AR parameters, differences, MA parameters,
        and periodicity. d must be an integer indicating the integration order of the process, while p and q may either
        be an integers indicating the AR and MA orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. s is an integer giving the periodicity (number of
        periods in season), often it is 4 for quarterly data or 12 for monthly data. Default is no seasonal effect.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the polynomial as in `numpy.poly1d`, where
        `[1,1,0,1]` would denote :math:`a + bt + ct^3`. Default is to not
        include a trend component.
    enforce_stationarity : boolean, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : boolean, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    method : str, optional (default='lbfgs')
        The method determines which solver from scipy.optimize is used, and it can be chosen from
        among the following strings:
        - ’newton’ for Newton-Raphson, ‘nm’ for Nelder-Mead
        - ’bfgs’ for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        - ’lbfgs’ for limited-memory BFGS with optional box constraints
        - ’powell’ for modified Powell’s method
        - ’cg’ for conjugate gradient
        - ’ncg’ for Newton-conjugate gradient
        - ’basinhopping’ for global basin-hopping solver
        The explicit arguments in fit are passed to the solver, with the exception of the basin-hopping solver.
        Each solver has several optional arguments that are not the same across solvers.
        See the notes section below (or scipy.optimize) for the available arguments and for the list of explicit
        arguments that the basin-hopping solver supports.
    maxiter : int, optional (default=1000)
        The maximum number of iterations to perfom in fitting the likelihood to the data.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """

    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend='n', enforce_stationarity=True,
                 enforce_invertibility=True, maxiter=1000, method='lbfgs', check_input=True, disp=0):
        # TODO add more constructor/fit options from statsmodels

        # Input checks.
        self._check_order(order, 3)
        self._check_order(seasonal_order, 4)

        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.method = method
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter
        self.disp = disp
        super(ARIMAForecaster, self).__init__(check_input=check_input)

    def _fit(self, y, X=None):
        """
        Internal fit.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must also be provided
            exogenous features for making predictions.

        Returns
        -------
        self : returns an instance of self.
        """
        # unnest series
        y = y.iloc[0]

        # fit estimator
        self._estimator = SARIMAX(y,
                                  exog=X,
                                  order=self.order,
                                  seasonal_order=self.seasonal_order,
                                  trend=self.trend,
                                  enforce_stationarity=self.enforce_stationarity,
                                  enforce_invertibility=self.enforce_invertibility)
        self._fitted_estimator = self._estimator.fit(maxiter=self.maxiter, method=self.method, disp=self.disp)
        return self

    def _update(self, y, X=None):
        """
        Internal update of forecasts using new data via Kalman smoothing/filtering of
        forecasts obtained from previously fitted forecaster.

        Parameters
        ----------
        y : pandas.Series
            Updated time series which to use for updating the previously fitted forecaster.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must also be provided
            exogenous features for making predictions.

        Returns
        -------
        self : An instance of self
        """
        # TODO for updating see https://github.com/statsmodels/statsmodels/issues/2788 and
        #  https://github.com/statsmodels/statsmodels/issues/3318

        # unnest series
        y = y.iloc[0]

        # Update estimator.
        estimator = SARIMAX(y,
                            exog=X,
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            trend=self.trend,
                            enforce_stationarity=self.enforce_stationarity,
                            enforce_invertibility=self.enforce_invertibility)
        estimator.initialize_known(self._fitted_estimator.predicted_state[:, -1],
                                   self._fitted_estimator.predicted_state_cov[:, :, -1])

        # Filter given fitted parameters.
        self._updated_estimator = estimator.smooth(self._fitted_estimator.params)
        return self

    def _predict(self, fh=None, X=None):
        """
        Internal predict using fitted estimator.

        Parameters
        ----------
        fh : array-like, optional (default=None)
            The forecasting horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that if
            provided, the forecaster must also have been fitted on the exogenous
            features.

        Returns
        -------
        Predictions : pandas.Series, shape=(len(fh),)
            Returns series of predicted values.
        """
        # Adjust forecasting horizon to time index seen in fit, (assume sorted forecasting horizon)
        fh = len(self._y_idx) - 1 + fh
        start = fh[0]
        end = fh[-1]

        # Predict updated (pre-initialised) model with start and end values relative to end of train series
        if self._is_updated:
            y_pred = self._updated_estimator.predict(start=start, end=end, exog=X)

        # Predict fitted model with start and end points relative to start of train series
        else:
            y_pred = self._fitted_estimator.predict(start=start, end=end, exog=X)

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        fh_idx = fh - np.min(fh)
        return y_pred.iloc[fh_idx]

    @staticmethod
    def _check_order(order, n):
        """Helper function to check passed order for ARIMA estimator.

        Parameters
        ----------
        order : tuple
            Estimator order
        n : int
            Length of order
        """
        if not (isinstance(order, tuple) and (len(order) == n)):
            raise ValueError(f'Order must be a tuple of length f{n}')

        if not all(np.issubdtype(type(k), np.integer) for k in order):
            raise ValueError(f'All values in order must be integers')


class ExpSmoothingForecaster(BaseSingleSeriesForecaster):
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
    seasonal_periods : int, optional (default=None)
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
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.

    References
    ----------
    [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
        and practice. OTexts, 2014.
    """

    def __init__(self, trend=None, damped=False, seasonal=None, seasonal_periods=None, smoothing_level=None,
                 smoothing_slope=None, smoothing_seasonal=None, damping_slope=None, optimized=True,
                 use_boxcox=False, remove_bias=False, use_basinhopping=False, check_input=True):

        # Model params
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        # Fit params
        self.smoothing_level = smoothing_level
        self.optimized = optimized
        self.smoothing_slope = smoothing_slope
        self.smooting_seasonal = smoothing_seasonal
        self.damping_slope = damping_slope
        self.use_boxcox = use_boxcox
        self.remove_bias = remove_bias
        self.use_basinhopping = use_basinhopping
        super(ExpSmoothingForecaster, self).__init__(check_input=check_input)

    def _fit(self, y):
        """
        Internal fit.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.

        Returns
        -------
        self : returns an instance of self.
        """

        # Unnest series.
        y = y.iloc[0]

        # Fit forecaster.
        self.estimator = ExponentialSmoothing(y, trend=self.trend, damped=self.damped, seasonal=self.seasonal,
                                              seasonal_periods=self.seasonal_periods)
        self._fitted_estimator = self.estimator.fit(smoothing_level=self.smoothing_level, optimized=self.optimized,
                                                    smoothing_slope=self.smoothing_slope,
                                                    smoothing_seasonal=self.smooting_seasonal, damping_slope=self.damping_slope,
                                                    use_boxcox=self.use_boxcox, remove_bias=self.remove_bias,
                                                    use_basinhopping=self.use_basinhopping)
        return self


class DummyForecaster(BaseSingleSeriesForecaster):
    """
    Dummy forecaster for naive forecasting approaches.

    Parameters
    ----------
    strategy : str{'mean', 'last', 'linear'}, optional (default='last')
        Naive forecasting strategy
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    def __init__(self, strategy='last', check_input=True):

        allowed_strategies = ('mean', 'last', 'linear') #seasonal_last
        if strategy not in allowed_strategies:
            raise ValueError(f'Unknown strategy: {strategy}, expected one of {allowed_strategies}')

        self.strategy = strategy
        super(DummyForecaster, self).__init__(check_input=check_input)

    def _fit(self, y):
        """
        Internal fit.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.

        Returns
        -------
        self : returns an instance of self.
        """
        # Unnest series.
        y = y.iloc[0]

        # Fit estimator.
        if self.strategy == 'mean':
            self.estimator = ExponentialSmoothing(y)
            self._fitted_estimator = self.estimator.fit(smoothing_level=0)

        if self.strategy == 'last':
            self.estimator = ExponentialSmoothing(y)
            self._fitted_estimator = self.estimator.fit(smoothing_level=1)

        if self.strategy == 'linear':
            self.estimator = SARIMAX(y, order=(0, 0, 0), trend='t')
            self._fitted_estimator = self.estimator.fit()

        return self


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble of forecasters.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples.
    weights : array-like, shape = [n_estimators], optional (default=None)
        Sequence of weights (float or int) to weight the occurrences of predicted values before averaging.
        Uses uniform weights if None.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    # TODO: experimental, major functionality not implemented (input checks, params interface, exogenous variables)

    def __init__(self, estimators=None, weights=None, check_input=True):
        # TODO add input checks
        self.estimators = estimators
        self.weights = weights
        self.fitted_estimators_ = []
        super(EnsembleForecaster, self).__init__(check_input=check_input)

    def _fit(self, y, X=None):
        """
        Internal fit.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must also be provided
            exogenous features for making predictions.

        Returns
        -------
        self : returns an instance of self.
        """
        for _, estimator in self.estimators:
            # TODO implement set/get params interface
            # estimator.set_params(**{"check_input": False})
            fitted_estimator = estimator.fit(y, X=X)
            self.fitted_estimators_.append(fitted_estimator)
        return self

    def _predict(self, fh=None, X=None):
        """
        Internal predict using fitted estimator.

        Parameters
        ----------
        fh : array-like, optional (default=None)
            The forecasting horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that if
            provided, the forecaster must also have been fitted on the exogenous
            features.

        Returns
        -------
        Predictions : pandas.Series, shape=(len(fh),)
            Returns series of predicted values.
        """
        # TODO pass X only to estimators where the predict method accepts X, currenlty X is ignored

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        fh_idx = fh - np.min(fh)

        # Iterate over estimators
        y_preds = np.zeros((len(self.fitted_estimators_), len(fh)))
        indexes = []
        for i, estimator in enumerate(self.fitted_estimators_):
            y_pred = estimator.predict(fh=fh)
            y_preds[i, :] = y_pred
            indexes.append(y_pred.index)

        # Check if all predicted horizons are identical
        if not all(index.equals(indexes[0]) for index in indexes):
            raise ValueError('Predicted horizons from estimators do not match')

        # Average predictions over estimators
        avg_preds = np.average(y_preds, axis=0, weights=self.weights)

        # Return average predictions with index
        index = indexes[0]
        name = y_preds[0].name if hasattr(y_preds[0], 'name') else None
        return pd.Series(avg_preds, index=index, name=name)

    def get_params(self, deep=True):
        # TODO fix get and set params interface following sklearn double underscore convention
        raise NotImplementedError()

    def set_params(self, **params):
        raise NotImplementedError()


