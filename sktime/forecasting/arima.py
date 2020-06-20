#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "AutoARIMA"
]

import numpy as np
import pandas as pd
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import BaseSktimeForecaster
from sktime.forecasting.base._sktime import OptionalForecastingHorizonMixin


class AutoARIMA(OptionalForecastingHorizonMixin, BaseSktimeForecaster):
    """Automatically discover the optimal order for an ARIMA model.

    The auto-ARIMA process seeks to identify the most optimal parameters
    for an ARIMA model, settling on a single fitted ARIMA model. This
    process is based on the commonly-used R function,
    forecast::auto.arima [3].

    Auto-ARIMA works by conducting differencing tests (i.e.,
    Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller or
    Phillips–Perron) to determine the order of differencing, d, and then
    fitting models within ranges of defined start_p, max_p, start_q, max_q
    ranges. If the seasonal optional is enabled, auto-ARIMA also seeks to
    identify the optimal P and Q hyper-parameters after conducting the
    Canova-Hansen to determine the optimal order of seasonal differencing, D.

    In order to find the best model, auto-ARIMA optimizes for a given
    information_criterion, one of (‘aic’, ‘aicc’, ‘bic’, ‘hqic’, ‘oob’)
    (Akaike Information Criterion, Corrected Akaike Information Criterion,
    Bayesian Information Criterion, Hannan-Quinn Information Criterion, or
    “out of bag”–for validation scoring–respectively) and returns the ARIMA
    which minimizes the value.

    Note that due to stationarity issues, auto-ARIMA might not find a suitable
    model that will converge. If this is the case, a ValueError will be thrown
    suggesting stationarity-inducing measures be taken prior to re-fitting or
    that a new range of order values be selected. Non- stepwise (i.e.,
    essentially a grid search) selection can be slow, especially for seasonal
    data. Stepwise algorithm is outlined in Hyndman and Khandakar (2008).

    Parameters
    ----------
    start_p : int, optional (default=2)
        The starting value of p, the order (or number of time lags)
            of the auto-regressive (“AR”) model. Must be a positive integer.
    d : int, optional
        The order of first-differencing. If None (by default), the value will
        automatically be selected based on the results of the test (i.e.,
        either the Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller
        or the Phillips–Perron test will be conducted to find the most probable
        value). Must be a positive integer or None. Note that if d is None,
        the runtime could be significantly longer., by default None
    start_q : int, optional
        The starting value of q, the order of the moving-average (“MA”) model.
        Must be a positive integer., by default 2
    max_p : int, optional
        The maximum value of p, inclusive. Must be a positive integer greater
        than or equal to start_p., by default 5
    max_d : int, optional
        The maximum value of d, or the maximum number of non-seasonal
        differences. Must be a positive integer greater than or equal to d.,
        by default 2
    max_q : int, optional
        he maximum value of q, inclusive. Must be a positive integer greater
        than start_q., by default 5
    start_P : int, optional
        The starting value of P, the order of the auto-regressive portion of
        the seasonal model., by default 1
    D : int, optional
        The order of the seasonal differencing. If None (by default, the value
        will automatically be selected based on the results of the
        seasonal_test. Must be a positive integer or None., by default None
    start_Q : int, optional
        The starting value of Q, the order of the moving-average portion of
        the seasonal model., by default 1
    max_P : int, optional
        The maximum value of P, inclusive. Must be a positive integer greater
        than start_P., by default 2
    max_D : int, optional
        The maximum value of D. Must be a positive integer greater than D.,
        by default 1
    max_Q : int, optional
        The maximum value of Q, inclusive. Must be a positive integer greater
        than start_Q., by default 2
    max_order : int, optional
        Maximum value of p+q+P+Q if model selection is not stepwise. If the
        sum of p and q is >= max_order, a model will not be fit with those
        parameters, but will progress to the next combination. Default is 5.
        If max_order is None, it means there are no constraints on maximum
        order., by default 5
    sp : int, optional
        The period for seasonal differencing, sp refers to the number of
        periods in each season. For example, sp is 4 for quarterly data, 12
        for monthly data, or 1 for annual (non-seasonal) data. Default is 1.
        Note that if sp == 1 (i.e., is non-seasonal), seasonal will be set to
        False. For more information on setting this parameter, see Setting sp.
        (link to http://alkaline-ml.com/pmdarima/tips_and_tricks.html#period),
        by default 1
    seasonal : bool, optional
        Whether to fit a seasonal ARIMA. Default is True. Note that if
        seasonal is True and m == 1, seasonal will be set to False., by
        default True
    stationary : bool, optional
        Whether the time-series is stationary and d should be set to zero., by
        default False
    information_criterion : str, optional
        The information criterion used to select the best ARIMA model. One of
        pmdarima.arima.auto_arima.VALID_CRITERIA, (‘aic’, ‘bic’, ‘hqic’, ‘oob’)
        , by default 'aic'
    alpha : float, optional
        Level of the test for testing significance., by default 0.05
    test : str, optional
        Type of unit root test to use in order to detect stationarity if
        stationary is False and d is None., by default 'kpss'
    seasonal_test : str, optional
        This determines which seasonal unit root test is used if seasonal is
        True and D is None., by default 'ocsb'
    stepwise : bool, optional
        Whether to use the stepwise algorithm outlined in Hyndman and
        Khandakar (2008) to identify the optimal model parameters. The
        stepwise algorithm can be significantly faster than fitting all (or a
        random subset of) hyper-parameter combinations and is less likely to
        over-fit the model., by default True
    n_jobs : int, optional
        The number of models to fit in parallel in the case of a grid search
        (stepwise=False). Default is 1, but -1 can be used to designate “as
        many as possible”., by default 1
    start_params : array-like, optional
        Starting parameters for ARMA(p,q). If None, the default is given by
        ARMA._fit_start_params., by default None
    trend : str, optional
        The trend parameter. If with_intercept is True, trend will be used. If
        with_intercept is False, the trend will be set to a no- intercept
        value., by default None
    method : str, optional
        The method determines which solver from scipy.optimize is used, and it
        can be chosen from among the following strings:

        ‘newton’ for Newton-Raphson
        ‘nm’ for Nelder-Mead
        ‘bfgs’ for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        ‘lbfgs’ for limited-memory BFGS with optional box constraints
        ‘powell’ for modified Powell’s method
        ‘cg’ for conjugate gradient
        ‘ncg’ for Newton-conjugate gradient
        ‘basinhopping’ for global basin-hopping solver
        The explicit arguments in fit are passed to the solver, with the
        exception of the basin-hopping solver. Each solver has several
        optional arguments that are not the same across solvers. These can be
        passed as **fit_kwargs, by default 'lbfgs'
                maxiter : int, optional
                    The maximum number of function evaluations., by default 50
                offset_test_args : dict, optional
                    The args to pass to the constructor of the offset (d) test.
                    See pmdarima.arima.stationarity for more details., by
                    default None
                seasonal_test_args : [type], optional
                    The args to pass to the constructor of the seasonal offset
                    (D) test. See pmdarima.arima.seasonality for more details.
                    , by default None
                suppress_warnings : bool, optional
                    Many warnings might be thrown inside of statsmodels. If
                    suppress_warnings is True, all of the warnings coming from
                    ARIMA will be squelched., by default False
                error_action : str, optional
                    If unable to fit an ARIMA due to stationarity issues,
                    whether to warn (‘warn’), raise the ValueError (‘raise’)
                    or ignore (‘ignore’). Note that the default behavior is to
                    warn, and fits that fail will be returned as None. This is
                    the recommended behavior, as statsmodels ARIMA and SARIMAX
                    models hit bugs periodically that can cause an otherwise
                    healthy parameter combination to fail for reasons not
                    related to pmdarima., by default 'warn'
                trace : bool, optional
                    Whether to print status on the fits. A value of False will
                    print no debugging information. A value of True will print
                    some. Integer values exceeding 1 will print increasing
                    amounts of debug information at each fit., by default False
                random : bool, optional
                    Similar to grid searches, auto_arima provides the
                    capability to perform a “random search” over a
                    hyper-parameter space. If random is True, rather than
                    perform an exhaustive search or stepwise search, only
                    n_fits ARIMA models will be fit (stepwise must be False
                    for this option to do anything)., by default False
                random_state : int, long or numpy RandomState, optional
                    The PRNG for when random=True. Ensures replicable testing
                    and results., by default None
                n_fits : int, optional
                    If random is True and a “random search” is going to be
                    performed, n_iter is the number of ARIMA models to be fit.
                    , by default 10
                out_of_sample_size : int, optional
                    The ARIMA class can fit only a portion of the data if
                    specified, in order to retain an “out of bag” sample score.
                    This is the number of examples from the tail of the time
                    series to hold out and use as validation examples. The
                    model will not be fit on these samples, but the
                    observations will be added into the model’s endog and exog
                    arrays so that future forecast values originate from the
                    end of the endogenous vector.

        # For instance:

        # y = [0, 1, 2, 3, 4, 5, 6]
        # out_of_sample_size = 2

        # > Fit on: [0, 1, 2, 3, 4]
        # > Score on: [5, 6]
        # > Append [5, 6] to end of self.arima_res_.data.endog values,
        # by default 0
    scoring : str, optional
        If performing validation (i.e., if out_of_sample_size > 0), the metric
        to use for scoring the out-of-sample data. One of (‘mse’, ‘mae’), by
        default 'mse'
    scoring_args : dict, optional
        A dictionary of key-word arguments to be passed to the scoring metric.,
        by default None
    with_intercept : bool, optional
        Whether to include an intercept term. Default is True., by default True
    """

    def __init__(self, start_p=2, d=None, start_q=2, max_p=5,
                 max_d=2, max_q=5, start_P=1, D=None, start_Q=1, max_P=2,
                 max_D=1, max_Q=2, max_order=5, sp=1, seasonal=True,
                 stationary=False, information_criterion='aic', alpha=0.05,
                 test='kpss', seasonal_test='ocsb', stepwise=True, n_jobs=1,
                 start_params=None, trend=None, method='lbfgs', maxiter=50,
                 offset_test_args=None, seasonal_test_args=None,
                 suppress_warnings=False, error_action='warn', trace=False,
                 random=False, random_state=None, n_fits=10,
                 out_of_sample_size=0, scoring='mse',
                 scoring_args=None, with_intercept=True,
                 **kwargs):

        self.start_p = start_p
        self.d = d
        self.start_q = start_q
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.start_P = start_P
        self.D = D
        self.start_Q = start_Q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.max_order = max_order
        self.sp = sp
        self.seasonal = seasonal
        self.stationary = stationary
        self.information_criterion = information_criterion
        self.alpha = alpha
        self.test = test
        self.seasonal_test = seasonal_test
        self.stepwise = stepwise
        self.n_jobs = n_jobs
        self.start_params = start_params
        self.trend = trend
        self.method = method
        self.maxiter = maxiter
        self.offset_test_args = offset_test_args
        self.seasonal_test_args = seasonal_test_args
        self.suppress_warnings = suppress_warnings
        self.error_action = error_action
        self.trace = trace
        self.random = random
        self.random_state = random_state
        self.n_fits = n_fits
        self.out_of_sample_size = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = scoring_args
        self.with_intercept = with_intercept

        super(AutoARIMA, self).__init__()

        # import inside method to avoid hard dependency
        from pmdarima.arima import AutoARIMA as _AutoARIMA
        self._forecaster = _AutoARIMA(
            start_p=start_p, d=d, start_q=start_q, max_p=max_p,
            max_d=max_d, max_q=max_q, start_P=start_P, D=D, start_Q=start_Q,
            max_P=max_P,
            max_D=max_D, max_Q=max_Q, max_order=max_order, m=sp,
            seasonal=seasonal,
            stationary=stationary, information_criterion=information_criterion,
            alpha=alpha,
            test=test, seasonal_test=seasonal_test, stepwise=stepwise,
            n_jobs=n_jobs,
            start_params=None, trend=trend, method=method, maxiter=maxiter,
            offset_test_args=offset_test_args,
            seasonal_test_args=seasonal_test_args,
            suppress_warnings=suppress_warnings, error_action=error_action,
            trace=trace,
            random=random, random_state=random_state, n_fits=n_fits,
            out_of_sample_size=out_of_sample_size, scoring=scoring,
            scoring_args=scoring_args, with_intercept=with_intercept,
            **kwargs
        )

    def fit(self, y_train, fh=None, X_train=None, **fit_args):
        """Fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X_train : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self._set_oh(y_train)
        self._set_fh(fh)
        self._forecaster.fit(y_train, exogenous=X_train, **fit_args)
        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        # distinguish between in-sample and out-of-sample prediction
        is_in_sample = fh <= 0
        is_out_of_sample = np.logical_not(is_in_sample)

        # pure out-of-sample prediction
        if np.all(is_out_of_sample):
            return self._predict_out_of_sample(fh, X=X,
                                               return_pred_int=return_pred_int,
                                               alpha=DEFAULT_ALPHA)

        # pure in-sample prediction
        elif np.all(is_in_sample):
            return self._predict_in_sample(fh, X=X,
                                           return_pred_int=return_pred_int,
                                           alpha=DEFAULT_ALPHA)

        # mixed in-sample and out-of-sample prediction
        else:
            fh_in_sample = fh[is_in_sample]
            fh_out_of_sample = fh[is_out_of_sample]

            y_pred_in = self._predict_in_sample(
                fh_in_sample, X=X,
                return_pred_int=return_pred_int,
                alpha=DEFAULT_ALPHA)
            y_pred_out = self._predict_out_of_sample(
                fh_out_of_sample, X=X,
                return_pred_int=return_pred_int,
                alpha=DEFAULT_ALPHA)
            return y_pred_in.append(y_pred_out)

    def _predict_in_sample(self, fh, X=None, return_pred_int=False,
                           alpha=DEFAULT_ALPHA):
        fh_abs = fh.absolute(self.cutoff)
        fh_idx = fh_abs - np.min(fh_abs)
        start = fh_abs[0]
        end = fh_abs[-1]

        if return_pred_int:

            if isinstance(alpha, (list, tuple)):
                raise NotImplementedError()
            y_pred, pred_int = self._forecaster.predict_in_sample(
                start=start,
                end=end,
                exogenous=X,
                return_conf_int=return_pred_int,
                alpha=alpha)
            y_pred = pd.Series(y_pred[fh_idx], index=fh_abs)
            pred_int = pd.DataFrame(pred_int[fh_idx, :], index=fh_abs,
                                    columns=["lower", "upper"])
            return y_pred, pred_int

        else:
            y_pred = self._forecaster.predict_in_sample(
                start=start, end=end,
                exogenous=X,
                return_conf_int=return_pred_int,
                alpha=alpha)
            return pd.Series(y_pred[fh_idx], index=fh_abs)

    def _predict_out_of_sample(self, fh, X=None, return_pred_int=False,
                               alpha=DEFAULT_ALPHA):
        # make prediction
        n_periods = int(fh[-1])
        index = fh.absolute(self.cutoff)
        fh_idx = fh.index_like(self.cutoff)

        if return_pred_int:
            y_pred, pred_int = self._forecaster.model_.predict(
                n_periods=n_periods, exogenous=X,
                return_conf_int=return_pred_int, alpha=alpha)
            y_pred = pd.Series(y_pred[fh_idx], index=index)
            pred_int = pd.DataFrame(pred_int[fh_idx, :], index=index,
                                    columns=["lower", "upper"])
            return y_pred, pred_int
        else:
            y_pred = self._forecaster.model_.predict(
                n_periods=n_periods,
                exogenous=X,
                return_conf_int=return_pred_int,
                alpha=alpha)
            return pd.Series(y_pred[fh_idx], index=index)

    def get_fitted_params(self):
        """Get fitted parameters

        Returns
        -------
        fitted_params : dict
        """
        self.check_is_fitted()
        names = self._get_fitted_param_names()
        params = self._forecaster.model_.arima_res_._results.params
        return {name: param for name, param in zip(names, params)}

    def _get_fitted_param_names(self):
        return self._forecaster.model_.arima_res_._results.param_names
