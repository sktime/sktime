#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)


"""Implements Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models and its variants."""

__author__ = ["Vasudeva-bit"]

from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("arch", severity="warning")

class ARCH(BaseForecaster):
    """Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model for forecasting 
    votility in high frequency time series data.

    Implements Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models introduced 
    and implemented in [1]_, [2]_, and [3]_ to forecast the volatility in high frequency time series data.
    
    ARCH models are a popular class of volatility models that use observed values of returns or residuals 
    as volatility shocks. A basic GARCH model is specified as
        rt = μ+ϵt
        ϵt = σtet
        σ^2t = ω+αϵ^2t−1+βσ^2t−1
    
    A complete ARCH model is divided into three components:
        a mean model, e.g., a constant mean or an ARX;
        a volatility process, e.g., a GARCH or an EGARCH process; and
        a distribution for the standardized residuals.

    Hyper-parameters
    ----------------
    y : {np.ndarray, Series, None}
        The dependent variable
    x : {np.ndarray, DataFrame}, optional
        Exogenous regressors.  Ignored if model does not permit exogenous
        regressors.
    mean : str, optional
        Name of the mean model.  Currently supported options are: 'Constant',
        'Zero', 'LS', 'AR', 'ARX', 'HAR' and  'HARX'
    lags : int or list (int), optional
        Either a scalar integer value indicating lag length or a list of
        integers specifying lag locations.
    vol : str, optional
        Name of the volatility model.  Currently supported options are:
        'GARCH' (default), 'ARCH', 'EGARCH', 'FIARCH' and 'HARCH'
    p : int, optional
        Lag order of the symmetric innovation
    o : int, optional
        Lag order of the asymmetric innovation
    q : int, optional
        Lag order of lagged volatility or equivalent
    power : float, optional
        Power to use with GARCH and related models
    dist : int, optional
        Name of the error distribution.  Currently supported options are:

            * Normal: 'normal', 'gaussian' (default)
            * Students's t: 't', 'studentst'
            * Skewed Student's t: 'skewstudent', 'skewt'
            * Generalized Error Distribution: 'ged', 'generalized error"

    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    rescale : bool
        Flag indicating whether to automatically rescale data if the scale
        of the data is likely to produce convergence issues when estimating
        model parameters. If False, the model is estimated on the data without
        transformation.  If True, than y is rescaled and the new scale is
        reported in the estimation results.

    See Also
    --------
    Autoregressive Integrated Moving Average (ARIMA) models

    References
    ----------
    .. [1] Jason Brownlee. How to Model Volatility with ARCH and GARCH for Time Series Forecasting.
       https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/
    .. [2] GitHub repository of arch package (dependency). 
       https://github.com/bashtage/arch 
    .. [3] Documentation of arch package (dependency). Forecasting Volatility with ARCH and its variants.
       https://arch.readthedocs.io/en/latest/univariate/introduction.html 

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.arch import ARCH
    >>> y = load_airline()
    >>> forecaster = ARCH(
    ...    mean='Constant', 
    ...    lags=0, vol='Garch', 
    ...    p=1, 
    ...    o=0, 
    ...    q=1, 
    ...    power=2.0, 
    ...    dist='Normal', 
    ...    hold_back=None, 
    ...    rescale=None)
    >>> forecaster.fit(y)
    ARCH(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "scitype:y": "univariate", 
        "y_inner_mtype": ("np.ndarray", "pd.Series"), 
        "X_inner_mtype": ("np.ndarray", "pd.DataFrame"),  
        "requires-fh-in-fit": False,  
        "capability:pred_int": False, 
    }


    def __init__(
        self, 
        mean='Constant', 
        lags=0, vol='Garch', 
        p=1, 
        o=0, 
        q=1, 
        power=2.0, 
        dist='Normal', 
        hold_back=None, 
        rescale=None):

        _check_soft_dependencies("arch", severity="error", object=self)

        self.mean = mean
        self.lags = lags
        self.vol = vol
        self.p = p
        self.o = o
        self.q = q
        self.power = power
        self.dist = dist
        self.hold_back = hold_back
        self.rescale = rescale
    
        super(ARCH, self).__init__()


    def _fit(self, y, X=None, **fit_params):
        """Fit the training data to the estimator

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        update_freq : int, optional
            Frequency of iteration updates.  Output is generated every
            update_freq iterations. Set to 0 to disable iterative output.
        disp : {bool, "off", "final"}
            Either 'final' to print optimization result or 'off' to display
            nothing. If using a boolean, False is "off" and True is "final"
        starting_values : np.ndarray, optional
            Array of starting values to use.  If not provided, starting values
            are constructed by the model components.
        cov_type : str, optional
            Estimation method of parameter covariance.  Supported options are
            'robust', which does not assume the Information Matrix Equality
            holds and 'classic' which does.  In the ARCH literature, 'robust'
            corresponds to Bollerslev-Wooldridge covariance estimator.
        show_warning : bool, optional
            Flag indicating whether convergence warnings should be shown
        first_obs : {int, str, datetime, Timestamp}
            First observation to use when estimating model
        last_obs : {int, str, datetime, Timestamp}
            Last observation to use when estimating model
        tol : float, optional
            Tolerance for termination.
        options : dict, optional
            Options to pass to scipy.optimize.minimize.  Valid entries
            include 'ftol', 'eps', 'disp', and 'maxiter'.
        backcast : {float, np.ndarray}, optional
            Value to use as backcast. Should be measure \sigma^2_0
            since model-specific non-linear transformations are applied to
            value before computing the variance recursions.

        Returns
        -------
        self : returns an instance of self
        """
        from arch import arch_model as _ARCH

        self.y = y
        self.X = X
        self._forecaster = _ARCH(
            y = self.y, 
            x = self.X, 
            mean = self.mean, 
            lags = self.lags, 
            vol = self.vol, 
            p = self.p,
            o = self.o, 
            q = self.q, 
            power = self.power, 
            dist = self.dist, 
            hold_back = self.hold_back, 
            rescale = self.rescale
            )

        self._forecaster.fit(**fit_params)
        return self


    def _predict(self, fh, X=None, **predict_params):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, 2d array, 3d-ndarray, 2d dictionay (dict) optional (default=None) 
        Exogenous time series.
   
        params : {np.ndarray, Series}
            Parameters required to forecast. Must be identical in shape to the parameters 
            computed by fitting the model.
        horizon : int, optional
            Number of steps to forecast
        start : {int, datetime, Timestamp, str}, optional
            An integer, datetime or str indicating the first observation to produce the 
            forecast for. Datetimes can only be used with pandas inputs that have a datetime 
            index. Strings must be convertible to a date time, such as in '1945-01-01'.
        align : str, optional
            Either 'origin' or 'target'. When set of 'origin', the t-th row of forecasts 
            contains the forecasts for t+1, t+2, ..., t+h. When set to 'target', the t-th 
            row contains the 1-step ahead forecast from time t-1, the 2 step from time t-2, ..., 
            and the h-step from time t-h. 'target' simplified computing forecast errors since 
            the realization and h-step forecast are aligned.
        method : {'analytic', 'simulation', 'bootstrap'}
            Method to use when producing the forecast. The default is analytic. The method only 
            affects the variance forecast generation. Not all volatility models support all methods. 
            In particular, volatility models that do not evolve in squares such as EGARCH or TARCH 
            do not support the 'analytic' method for horizons > 1.
        simulations : int
            Number of simulations to run when computing the forecast using either simulation or 
            bootstrap.
        rng : callable, optional
            Custom random number generator to use in simulation-based forecasts. Must produce random 
            samples using the syntax rng(size) where size the 2-element tuple (simulations, horizon).
        random_state : RandomState, optional
            NumPy RandomState instance to use when method is 'bootstrap'
        reindex : bool, optional
            Whether to reindex the forecasts to have the same dimension as the series being forecast. 
            Prior to 4.18 this was the default. As of 4.19 this is now optional. If not provided, a 
            warning is raised about the future change in the default which will occur after September 2021.
        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        y_pred = self._forecaster.forecast(horizon = fh, **predict_params)
        return y_pred

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        self.check_is_fitted()
        fitted_params = {}
        for name in self._get_fitted_param_names():
            fitted_params[name] = getattr(self._forecaster, name, None)
        return fitted_params

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict
        """
        params = {
            "mean" : 'Constant', 
            "lags" : 0, 
            "vol" : 'Garch', 
            "p" : 1, 
            "o" : 0, 
            "q" : 1, 
            "power" : 2.0, 
            "dist" : 'Normal', 
            "hold_back" : None, 
            "rescale" : None,
        }
        return params

    def summary(self):
        """Summary of the fitted model."""
        return self._forecaster.summary()