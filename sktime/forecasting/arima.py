#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["AutoARIMA"]

import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.utils.check_imports import _check_soft_dependencies

_check_soft_dependencies("pmdarima")


class AutoARIMA(_OptionalForecastingHorizonMixin, _SktimeForecaster):
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
    d : int, optional (default=None)
        The order of first-differencing. If None (by default), the value will
        automatically be selected based on the results of the test (i.e.,
        either the Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller
        or the Phillips–Perron test will be conducted to find the most probable
        value). Must be a positive integer or None. Note that if d is None,
        the runtime could be significantly longer.
    start_q : int, optional (default=2)
        The starting value of q, the order of the moving-average (“MA”) model.
        Must be a positive integer.
    max_p : int, optional (default=5)
        The maximum value of p, inclusive. Must be a positive integer greater
        than or equal to start_p.
    max_d : int, optional (default=2)
        The maximum value of d, or the maximum number of non-seasonal
        differences. Must be a positive integer greater than or equal to d.
    max_q : int, optional (default=5)
        he maximum value of q, inclusive. Must be a positive integer greater
        than start_q.
    start_P : int, optional (default=1)
        The starting value of P, the order of the auto-regressive portion of
        the seasonal model.
    D : int, optional (default=None)
        The order of the seasonal differencing. If None (by default, the value
        will automatically be selected based on the results of the
        seasonal_test. Must be a positive integer or None.
    start_Q : int, optional (default=1)
        The starting value of Q, the order of the moving-average portion of
        the seasonal model.
    max_P : int, optional (default=2)
        The maximum value of P, inclusive. Must be a positive integer greater
        than start_P.
    max_D : int, optional (default=1)
        The maximum value of D. Must be a positive integer greater than D.
    max_Q : int, optional (default=2)
        The maximum value of Q, inclusive. Must be a positive integer greater
        than start_Q.
    max_order : int, optional (default=5)
        Maximum value of p+q+P+Q if model selection is not stepwise. If the
        sum of p and q is >= max_order, a model will not be fit with those
        parameters, but will progress to the next combination. Default is 5.
        If max_order is None, it means there are no constraints on maximum
        order.
    sp : int, optional (default=1)
        The period for seasonal differencing, sp refers to the number of
        periods in each season. For example, sp is 4 for quarterly data, 12
        for monthly data, or 1 for annual (non-seasonal) data. Default is 1.
        Note that if sp == 1 (i.e., is non-seasonal), seasonal will be set to
        False. For more information on setting this parameter, see Setting sp.
        (link to http://alkaline-ml.com/pmdarima/tips_and_tricks.html#period)
    seasonal : bool, optional (default=True)
        Whether to fit a seasonal ARIMA. Default is True. Note that if
        seasonal is True and sp == 1, seasonal will be set to False.
    stationary : bool, optional (default=False)
        Whether the time-series is stationary and d should be set to zero.
    information_criterion : str, optional (default='aic')
        The information criterion used to select the best ARIMA model. One of
        pmdarima.arima.auto_arima.VALID_CRITERIA, (‘aic’, ‘bic’, ‘hqic’,
        ‘oob’).
    alpha : float, optional (default=0.05)
        Level of the test for testing significance.
    test : str, optional (default='kpss')
        Type of unit root test to use in order to detect stationarity if
        stationary is False and d is None.
    seasonal_test : str, optional (default='ocsb')
        This determines which seasonal unit root test is used if seasonal is
        True and D is None.
    stepwise : bool, optional (default=True)
        Whether to use the stepwise algorithm outlined in Hyndman and
        Khandakar (2008) to identify the optimal model parameters. The
        stepwise algorithm can be significantly faster than fitting all (or a
        random subset of) hyper-parameter combinations and is less likely to
        over-fit the model.
    n_jobs : int, optional (default=1)
        The number of models to fit in parallel in the case of a grid search
        (stepwise=False). Default is 1, but -1 can be used to designate “as
        many as possible”.
    start_params : array-like, optional (default=None)
        Starting parameters for ARMA(p,q). If None, the default is given by
        ARMA._fit_start_params.
    trend : str, optional (default=None)
        The trend parameter. If with_intercept is True, trend will be used. If
        with_intercept is False, the trend will be set to a no- intercept
        value.
    method : str, optional (default='lbfgs')
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
        passed as **fit_kwargs
    maxiter : int, optional (default=50)
        The maximum number of function evaluations.
    offset_test_args : dict, optional (default=None)
        The args to pass to the constructor of the offset (d) test.
        See pmdarima.arima.stationarity for more details.
    seasonal_test_args : dict, optional (default=None)
        The args to pass to the constructor of the seasonal offset (D) test.
        See pmdarima.arima.seasonality for more details.
    suppress_warnings : bool, optional (default=False)
        Many warnings might be thrown inside of statsmodels. If
        suppress_warnings is True, all of the warnings coming from ARIMA will
        be squelched.
    error_action : str, optional (default='warn')
        If unable to fit an ARIMA due to stationarity issues, whether to warn
        (‘warn’), raise the ValueError (‘raise’) or ignore (‘ignore’). Note
        that the default behavior is to warn, and fits that fail will be
        returned as None. This is the recommended behavior, as statsmodels
        ARIMA and SARIMAX models hit bugs periodically that can cause an
        otherwise healthy parameter combination to fail for reasons not
        related to pmdarima.
    trace : bool, optional (default=False)
        Whether to print status on the fits. A value of False will print no
        debugging information. A value of True will print some. Integer values
        exceeding 1 will print increasing amounts of debug information at each
        fit.
    random : bool, optional (default='False')
        Similar to grid searches, auto_arima provides the capability to
        perform a “random search” over a hyper-parameter space. If random is
        True, rather than perform an exhaustive search or stepwise search,
        only n_fits ARIMA models will be fit (stepwise must be False for this
        option to do anything).
    random_state : int, long or numpy RandomState, optional (default=None)
        The PRNG for when random=True. Ensures replicable testing and results.
    n_fits : int, optional (default=10)
        If random is True and a “random search” is going to be performed,
        n_iter is the number of ARIMA models to be fit.
    out_of_sample_size : int, optional (default=0)
        The ARIMA class can fit only a portion of the data if specified, in
        order to retain an “out of bag” sample score. This is the number of
        examples from the tail of the time series to hold out and use as
        validation examples. The model will not be fit on these samples, but
        the observations will be added into the model’s endog and exog arrays
        so that future forecast values originate from the end of the
        endogenous vector.

        # For instance:

        # y = [0, 1, 2, 3, 4, 5, 6]
        # out_of_sample_size = 2

        # > Fit on: [0, 1, 2, 3, 4]
        # > Score on: [5, 6]
        # > Append [5, 6] to end of self.arima_res_.data.endog values,
    scoring : str, optional (default='mse')
        If performing validation (i.e., if out_of_sample_size > 0), the metric
        to use for scoring the out-of-sample data. One of (‘mse’, ‘mae’)
    scoring_args : dict, optional (default=None)
        A dictionary of key-word arguments to be passed to the scoring metric.
    with_intercept : bool, optional (default=True)
        Whether to include an intercept term.

    References
    ----------
    https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html
    """

    def __init__(
        self,
        start_p=2,
        d=None,
        start_q=2,
        max_p=5,
        max_d=2,
        max_q=5,
        start_P=1,
        D=None,
        start_Q=1,
        max_P=2,
        max_D=1,
        max_Q=2,
        max_order=5,
        sp=1,
        seasonal=True,
        stationary=False,
        information_criterion="aic",
        alpha=0.05,
        test="kpss",
        seasonal_test="ocsb",
        stepwise=True,
        n_jobs=1,
        start_params=None,
        trend=None,
        method="lbfgs",
        maxiter=50,
        offset_test_args=None,
        seasonal_test_args=None,
        suppress_warnings=False,
        error_action="warn",
        trace=False,
        random=False,
        random_state=None,
        n_fits=10,
        out_of_sample_size=0,
        scoring="mse",
        scoring_args=None,
        with_intercept=True,
        **kwargs
    ):

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
            start_p=start_p,
            d=d,
            start_q=start_q,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            start_P=start_P,
            D=D,
            start_Q=start_Q,
            max_P=max_P,
            max_D=max_D,
            max_Q=max_Q,
            max_order=max_order,
            m=sp,
            seasonal=seasonal,
            stationary=stationary,
            information_criterion=information_criterion,
            alpha=alpha,
            test=test,
            seasonal_test=seasonal_test,
            stepwise=stepwise,
            n_jobs=n_jobs,
            start_params=None,
            trend=trend,
            method=method,
            maxiter=maxiter,
            offset_test_args=offset_test_args,
            seasonal_test_args=seasonal_test_args,
            suppress_warnings=suppress_warnings,
            error_action=error_action,
            trace=trace,
            random=random,
            random_state=random_state,
            n_fits=n_fits,
            out_of_sample_size=out_of_sample_size,
            scoring=scoring,
            scoring_args=scoring_args,
            with_intercept=with_intercept,
            **kwargs
        )

    def fit(self, y, X=None, fh=None, **fit_params):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self._set_y_X(y, X)
        self._set_fh(fh)
        self._forecaster.fit(y, X, **fit_params)
        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        # distinguish between in-sample and out-of-sample prediction
        fh_oos = fh.to_out_of_sample(self.cutoff)
        fh_ins = fh.to_in_sample(self.cutoff)

        kwargs = {"X": X, "return_pred_int": return_pred_int, "alpha": alpha}

        # all values are out-of-sample
        if len(fh_oos) == len(fh):
            return self._predict_fixed_cutoff(fh_oos, **kwargs)

        # all values are in-sample
        elif len(fh_ins) == len(fh):
            return self._predict_in_sample(fh_ins, **kwargs)

        # both in-sample and out-of-sample values
        else:
            y_ins = self._predict_in_sample(fh_ins, **kwargs)
            y_oos = self._predict_fixed_cutoff(fh_oos, **kwargs)
            return y_ins.append(y_oos)

    def _predict_in_sample(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        if isinstance(alpha, (list, tuple)):
            raise NotImplementedError()

        # for in-sample predictions, pmdarima requires zero-based
        # integer indicies
        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]

        result = self._forecaster.predict_in_sample(
            start=start,
            end=end,
            exogenous=X,
            return_conf_int=return_pred_int,
            alpha=alpha,
        )

        fh_abs = fh.to_absolute(self.cutoff)
        fh_idx = fh.to_indexer(self.cutoff, from_cutoff=False)

        if return_pred_int:
            # unpack and format results
            y_pred, pred_int = result
            y_pred = pd.Series(y_pred[fh_idx], index=fh_abs)
            pred_int = pd.DataFrame(
                pred_int[fh_idx, :], index=fh_abs, columns=["lower", "upper"]
            )
            return y_pred, pred_int

        else:
            return pd.Series(result[fh_idx], index=fh_abs)

    def _predict_fixed_cutoff(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # make prediction
        n_periods = int(fh.to_relative(self.cutoff)[-1])
        fh_abs = fh.to_absolute(self.cutoff)
        fh_idx = fh.to_indexer(self.cutoff)

        result = self._forecaster.predict(
            n_periods=n_periods,
            exogenous=X,
            return_conf_int=return_pred_int,
            alpha=alpha,
        )

        if return_pred_int:
            y_pred, pred_int = result
            y_pred = pd.Series(y_pred[fh_idx], index=fh_abs)
            pred_int = pd.DataFrame(
                pred_int[fh_idx, :], index=fh_abs, columns=["lower", "upper"]
            )
            return y_pred, pred_int
        else:
            return pd.Series(result[fh_idx], index=fh_abs)

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
