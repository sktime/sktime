#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface to ARIMA and AutoARIMA models from pmdarima package."""

__author__ = ["mloning", "hyang1996", "fkiraly", "ilkersigirci"]
__all__ = ["AutoARIMA", "ARIMA"]

from sktime.forecasting.base.adapters._pmdarima import _PmdArimaAdapter


class AutoARIMA(_PmdArimaAdapter):
    """Auto-(S)ARIMA(X) forecaster, from pmdarima package.

    Includes automated fitting of (S)ARIMA(X) hyper-parameters (p, d, q, P, D, Q).

    Exposes ``pmdarima.arima.AutoARIMA`` [1]_ under the ``sktime`` interface.
    Seasonal ARIMA models and exogeneous input is supported, hence this estimator is
    capable of fitting auto-SARIMA, auto-ARIMAX, and auto-SARIMAX.

    The auto-ARIMA algorithm seeks to identify the most optimal parameters
    for an ARIMA model, settling on a single fitted ARIMA model. This
    process is based on the commonly-used R function,
    forecast::auto.arima.

    Auto-ARIMA works by conducting differencing tests (i.e.,
    Kwiatkowski-Phillips-Schmidt-Shin, Augmented Dickey-Fuller or
    Phillips-Perron) to determine the order of differencing, d, and then
    fitting models within ranges of defined start_p, max_p, start_q, max_q
    ranges. If the seasonal optional is enabled, auto-ARIMA also seeks to
    identify the optimal P and Q hyper-parameters after conducting the
    Canova-Hansen to determine the optimal order of seasonal differencing, D.

    In order to find the best model, auto-ARIMA optimizes for a given
    information_criterion, one of ('aic', 'aicc', 'bic', 'hqic', 'oob')
    (Akaike Information Criterion, Corrected Akaike Information Criterion,
    Bayesian Information Criterion, Hannan-Quinn Information Criterion, or
    "out of bag"-for validation scoring-respectively) and returns the ARIMA
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
        of the auto-regressive ("AR") model. Must be a positive integer.
    d : int, optional (default=None)
        The order of first-differencing. If None (by default), the value will
        automatically be selected based on the results of the test (i.e.,
        either the Kwiatkowski-Phillips-Schmidt-Shin, Augmented Dickey-Fuller
        or the Phillips-Perron test will be conducted to find the most probable
        value). Must be a positive integer or None. Note that if d is None,
        the runtime could be significantly longer.
    start_q : int, optional (default=2)
        The starting value of q, the order of the moving-average ("MA") model.
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
        pmdarima.arima.auto_arima.VALID_CRITERIA, ('aic', 'bic', 'hqic',
        'oob').
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
        (stepwise=False). Default is 1, but -1 can be used to designate "as
        many as possible".
    start_params : array-like, optional (default=None)
        Starting parameters for ARMA(p,q). If None, the default is given by
        ARMA._fit_start_params.
    trend : str, optional (default=None)
        The trend parameter. If with_intercept is True, trend will be used. If
        with_intercept is False, the trend will be set to a no- intercept
        value.
    method : str, optional (default='lbfgs')
        The ``method`` determines which solver from ``scipy.optimize``
        is used, and it can be chosen from among the following strings:

        - 'newton' for Newton-Raphson
        - 'nm' for Nelder-Mead
        - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        - 'lbfgs' for limited-memory BFGS with optional box constraints
        - 'powell' for modified Powell's method
        - 'cg' for conjugate gradient
        - 'ncg' for Newton-conjugate gradient
        - 'basinhopping' for global basin-hopping solver

        The explicit arguments in ``fit`` are passed to the solver,
        with the exception of the basin-hopping solver. Each
        solver has several optional arguments that are not the same across
        solvers. These can be passed as **fit_kwargs
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
        ('warn'), raise the ValueError ('raise') or ignore ('ignore'). Note
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
        perform a "random search" over a hyper-parameter space. If random is
        True, rather than perform an exhaustive search or stepwise search,
        only n_fits ARIMA models will be fit (stepwise must be False for this
        option to do anything).
    random_state : int, long or numpy RandomState, optional (default=None)
        The PRNG for when random=True. Ensures replicable testing and results.
    n_fits : int, optional (default=10)
        If random is True and a "random search" is going to be performed,
        n_iter is the number of ARIMA models to be fit.
    out_of_sample_size : int, optional (default=0)
        The number of examples from the tail of the time series to hold out
        and use as validation examples. The model will not be fit on these
        samples, but the observations will be added into the model's ``endog``
        and ``exog`` arrays so that future forecast values originate from the
        end of the endogenous vector. For instance::
            y = [0, 1, 2, 3, 4, 5, 6]
            out_of_sample_size = 2
            > Fit on: [0, 1, 2, 3, 4]
            > Score on: [5, 6]
            > Append [5, 6] to end of self.arima_res_.data.endog values
    scoring : str, optional (default='mse')
        If performing validation (i.e., if out_of_sample_size > 0), the metric
        to use for scoring the out-of-sample data. One of ('mse', 'mae')
    scoring_args : dict, optional (default=None)
        A dictionary of key-word arguments to be passed to the scoring metric.
    with_intercept : bool, optional (default=True)
        Whether to include an intercept term.
    update_pdq : bool, optional (default=True)
        whether to update pdq parameters in update
        True: model is refit on all data seen so far, potentially updating p,d,q
        False: model updates only ARIMA coefficients via likelihood, as in pmdarima
    Further arguments to pass to the SARIMAX constructor:
    - time_varying_regression : boolean, optional (default=False)
        Whether or not coefficients on the exogenous regressors are allowed
        to vary over time.
    - enforce_stationarity : boolean, optional (default=True)
        Whether or not to transform the AR parameters to enforce
        stationarity in the auto-regressive component of the model.
        - enforce_invertibility : boolean, optional (default=True)
        Whether or not to transform the MA parameters to enforce
        invertibility in the moving average component of the model.
    - simple_differencing : boolean, optional (default=False)
        Whether or not to use partially conditional maximum likelihood
        estimation for seasonal ARIMA models. If True, differencing is
        performed prior to estimation, which discards the first
        :math:`s D + d` initial rows but results in a smaller
        state-space formulation. If False, the full SARIMAX model is
        put in state-space form so that all datapoints can be used in
        estimation. Default is False.
    - measurement_error: boolean, optional (default=False)
        Whether or not to assume the endogenous observations endog were
        measured with error. Default is False.
    - mle_regression : boolean, optional (default=True)
        Whether or not to use estimate the regression coefficients for the
        exogenous variables as part of maximum likelihood estimation or
        through the Kalman filter (i.e. recursive least squares). If
        time_varying_regression is True, this must be set to False.
        Default is True.
    - hamilton_representation : boolean, optional (default=False)
        Whether or not to use the Hamilton representation of an ARMA
        process (if True) or the Harvey representation (if False).
        Default is False.
    - concentrate_scale : boolean, optional (default=False)
        Whether or not to concentrate the scale (variance of the error
        term) out of the likelihood. This reduces the number of parameters
        estimated by maximum likelihood by one, but standard errors will
        then not be available for the scale parameter.

    See Also
    --------
    ARIMA
    StatsForecastAutoARIMA

    References
    ----------
    .. [1]
    https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.arima import AutoARIMA
    >>> y = load_airline()
    >>> forecaster = AutoARIMA(
    ...     sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    AutoARIMA(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "tgsmith61591",  # for pmdarima
            "charlesdrotar",  # for pmdarima
            "aaronreidsmith",  # for pmdarima
            "mloning",
            "hyang1996",
            "fkiraly",
            "ilkersigirci",
        ],
        "maintainers": ["hyang1996"],
        # python_dependencies: "pmdarima" - inherited from _PmdArimaAdapter
        # estimator type
        # --------------
        "handles-missing-data": True,
    }

    SARIMAX_KWARGS_KEYS = [
        "time_varying_regression",
        "enforce_stationarity",
        "enforce_invertibility",
        "simple_differencing",
        "measurement_error",
        "mle_regression",
        "hamilton_representation",
        "concentrate_scale",
    ]

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
        update_pdq=True,
        time_varying_regression=False,
        enforce_stationarity=True,
        enforce_invertibility=True,
        simple_differencing=False,
        measurement_error=False,
        mle_regression=True,
        hamilton_representation=False,
        concentrate_scale=False,
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
        self.update_pdq = update_pdq
        for key in self.SARIMAX_KWARGS_KEYS:
            setattr(self, key, eval(key))

        super().__init__()

        self._sp = sp if sp else 1

    def _instantiate_model(self):
        # import inside method to avoid hard dependency
        from pmdarima.arima import AutoARIMA as _AutoARIMA  # type: ignore

        sarimax_kwargs = {key: getattr(self, key) for key in self.SARIMAX_KWARGS_KEYS}

        return _AutoARIMA(
            start_p=self.start_p,
            d=self.d,
            start_q=self.start_q,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            start_P=self.start_P,
            D=self.D,
            start_Q=self.start_Q,
            max_P=self.max_P,
            max_D=self.max_D,
            max_Q=self.max_Q,
            max_order=self.max_order,
            m=self._sp,
            seasonal=self.seasonal,
            stationary=self.stationary,
            information_criterion=self.information_criterion,
            alpha=self.alpha,
            test=self.test,
            seasonal_test=self.seasonal_test,
            stepwise=self.stepwise,
            n_jobs=self.n_jobs,
            start_params=None,
            trend=self.trend,
            method=self.method,
            maxiter=self.maxiter,
            offset_test_args=self.offset_test_args,
            seasonal_test_args=self.seasonal_test_args,
            suppress_warnings=self.suppress_warnings,
            error_action=self.error_action,
            trace=self.trace,
            random=self.random,
            random_state=self.random_state,
            n_fits=self.n_fits,
            out_of_sample_size=self.out_of_sample_size,
            scoring=self.scoring,
            scoring_args=self.scoring_args,
            with_intercept=self.with_intercept,
            **sarimax_kwargs,
        )

    def _update(self, y, X=None, update_params=True):
        """Update model with data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        update_pdq = self.update_pdq
        if update_params:
            if update_pdq:
                self._fit(y=self._y, X=self._X, fh=self._fh)
            else:
                if X is not None:
                    X = X.loc[y.index]
                self._forecaster.update(y=y, X=X)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params = {
            "d": 0,
            "suppress_warnings": True,
            "max_p": 2,
            "max_q": 2,
            "seasonal": False,
        }
        params2 = {
            "d": 0,
            "suppress_warnings": True,
            "max_p": 2,
            "max_q": 2,
            "seasonal": False,
            "update_pdq": True,
        }
        return [params, params2]


class ARIMA(_PmdArimaAdapter):
    """(S)ARIMA(X) forecaster, from pmdarima package.

    Exposes ``pmdarima.arima.ARIMA`` [1]_ under the ``sktime`` interface.
    Seasonal ARIMA models and exogeneous input is supported, hence this estimator is
    capable of fitting SARIMA, ARIMAX, and SARIMAX.
    To additionally fit (S)ARIMA(X) hyper-parameters, use the ``AutoARIMA`` estimator.

    An ARIMA, or autoregressive integrated moving average model, is a
    generalization of an autoregressive moving average (ARMA) model, and is fitted to
    time-series data in an effort to forecast future points. ARIMA models can
    be especially efficacious in cases where data shows evidence of
    non-stationarity.

    The "AR" part of ARIMA indicates that the evolving variable of interest is
    regressed on its own lagged (i.e., prior observed) values. The "MA" part
    indicates that the regression error is actually a linear combination of
    error terms whose values occurred contemporaneously and at various times
    in the past. The "I" (for "integrated") indicates that the data values
    have been replaced with the difference between their values and the
    previous values (and this differencing process may have been performed
    more than once). The purpose of each of these features is to make the model
    fit the data as well as possible.

    Non-seasonal ARIMA models are generally denoted ``ARIMA(p,d,q)`` where
    parameters ``p``, ``d``, and ``q`` are non-negative integers, ``p`` is the
    order (number of time lags) of the autoregressive model, ``d`` is the
    degree of differencing (the number of times the data have had past values
    subtracted), and ``q`` is the order of the moving-average model. Seasonal
    ARIMA models are usually denoted ``ARIMA(p,d,q)(P,D,Q)m``, where ``m``
    refers to the number of periods in each season, and the uppercase ``P``,
    ``D``, ``Q`` refer to the autoregressive, differencing, and moving average
    terms for the seasonal part of the ARIMA model.

    When two out of the three terms are zeros, the model may be referred to
    based on the non-zero parameter, dropping "AR", "I" or "MA" from the
    acronym describing the model. For example, ``ARIMA(1,0,0)`` is ``AR(1)``,
    ``ARIMA(0,1,0)`` is ``I(1)``, and ``ARIMA(0,0,1)`` is ``MA(1)``.

    See notes for more practical information on the ``ARIMA`` class.

    Parameters
    ----------
    order : iterable or array-like, shape=(3,), optional (default=(1, 0, 0))
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters to use. ``p`` is the order (number of
        time lags) of the auto-regressive model, and is a non-negative integer.
        ``d`` is the degree of differencing (the number of times the data have
        had past values subtracted), and is a non-negative integer. ``q`` is
        the order of the moving-average model, and is a non-negative integer.
        Default is an AR(1) model: (1,0,0).
    seasonal_order : array-like, shape=(4,), optional (default=(0, 0, 0, 0))
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. ``D`` must
        be an integer indicating the integration order of the process, while
        ``P`` and ``Q`` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. ``S`` is an
        integer giving the periodicity (number of periods in season), often it
        is 4 for quarterly data or 12 for monthly data. Default is no seasonal
        effect.
    start_params : array-like, optional (default=None)
        Starting parameters for ``ARMA(p,q)``.  If None, the default is given
        by ``ARMA._fit_start_params``.
    method : str, optional (default='lbfgs')
        The ``method`` determines which solver from ``scipy.optimize``
        is used, and it can be chosen from among the following strings:

        - 'newton' for Newton-Raphson
        - 'nm' for Nelder-Mead
        - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        - 'lbfgs' for limited-memory BFGS with optional box constraints
        - 'powell' for modified Powell's method
        - 'cg' for conjugate gradient
        - 'ncg' for Newton-conjugate gradient
        - 'basinhopping' for global basin-hopping solver

        The explicit arguments in ``fit`` are passed to the solver,
        with the exception of the basin-hopping solver. Each
        solver has several optional arguments that are not the same across
        solvers. These can be passed as **fit_kwargs
    maxiter : int, optional (default=50)
        The maximum number of function evaluations. Default is 50
    suppress_warnings : bool, optional (default=False)
        Many warnings might be thrown inside of statsmodels. If
        ``suppress_warnings`` is True, all of these warnings will be squelched.
    out_of_sample_size : int, optional (default=0)
        The number of examples from the tail of the time series to hold out
        and use as validation examples. The model will not be fit on these
        samples, but the observations will be added into the model's ``endog``
        and ``exog`` arrays so that future forecast values originate from the
        end of the endogenous vector. See :func:`update`.
        For instance::
            y = [0, 1, 2, 3, 4, 5, 6]
            out_of_sample_size = 2
            > Fit on: [0, 1, 2, 3, 4]
            > Score on: [5, 6]
            > Append [5, 6] to end of self.arima_res_.data.endog values
    scoring : str or callable, optional (default='mse')
        If performing validation (i.e., if ``out_of_sample_size`` > 0), the
        metric to use for scoring the out-of-sample data:

        - If a string, must be a valid metric name importable from
          ``sklearn.metrics``

        - If a callable, must adhere to the function signature::

            def foo_loss(y_true, y_pred)

        Note that models are selected by *minimizing* loss. If using a
        maximizing metric (such as ``sklearn.metrics.r2_score``), it is the
        user's responsibility to wrap the function such that it returns a
        negative value for minimizing.
    scoring_args : dict, optional (default=None)
        A dictionary of key-word arguments to be passed to the
        ``scoring`` metric.
    trend : str or None, optional (default=None)
        The trend parameter. If ``with_intercept`` is True, ``trend`` will be
        used. If ``with_intercept`` is False, the trend will be set to a no-
        intercept value. If None and ``with_intercept``, 'c' will be used as
        a default.
    with_intercept : bool, optional (default=True)
        Whether to include an intercept term. Default is True.
    Further arguments to pass to the SARIMAX constructor:
    - time_varying_regression : boolean, optional (default=False)
        Whether or not coefficients on the exogenous regressors are allowed
        to vary over time.
    - enforce_stationarity : boolean, optional (default=True)
        Whether or not to transform the AR parameters to enforce
        stationarity in the auto-regressive component of the model.
        - enforce_invertibility : boolean, optional (default=True)
        Whether or not to transform the MA parameters to enforce
        invertibility in the moving average component of the model.
    - simple_differencing : boolean, optional (default=False)
        Whether or not to use partially conditional maximum likelihood
        estimation for seasonal ARIMA models. If True, differencing is
        performed prior to estimation, which discards the first
        :math:`s D + d` initial rows but results in a smaller
        state-space formulation. If False, the full SARIMAX model is
        put in state-space form so that all datapoints can be used in
        estimation. Default is False.
    - measurement_error: boolean, optional (default=False)
        Whether or not to assume the endogenous observations endog were
        measured with error. Default is False.
    - mle_regression : boolean, optional (default=True)
        Whether or not to use estimate the regression coefficients for the
        exogenous variables as part of maximum likelihood estimation or
        through the Kalman filter (i.e. recursive least squares). If
        time_varying_regression is True, this must be set to False.
        Default is True.
    - hamilton_representation : boolean, optional (default=False)
        Whether or not to use the Hamilton representation of an ARMA
        process (if True) or the Harvey representation (if False).
        Default is False.
    - concentrate_scale : boolean, optional (default=False)
        Whether or not to concentrate the scale (variance of the error
        term) out of the likelihood. This reduces the number of parameters
        estimated by maximum likelihood by one, but standard errors will
        then not be available for the scale parameter.

    See Also
    --------
    AutoARIMA

    References
    ----------
    .. [1] https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html

    .. [2]
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.arima import ARIMA
    >>> y = load_airline()
    >>> forecaster = ARIMA(  # doctest: +SKIP
    ...     order=(1, 1, 0),
    ...     seasonal_order=(0, 1, 0, 12),
    ...     suppress_warnings=True)
    >>> forecaster.fit(y)  # doctest: +SKIP
    ARIMA(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """  # noqa: E501

    _tags = {
        "authors": [
            "tgsmith61591",  # for pmdarima
            "charlesdrotar",  # for pmdarima
            "aaronreidsmith",  # for pmdarima
            "mloning",
            "hyang1996",
            "fkiraly",
            "ilkersigirci",
        ],
        "maintainers": ["hyang1996"],
        "handles-missing-data": True,
    }

    SARIMAX_KWARGS_KEYS = [
        "time_varying_regression",
        "enforce_stationarity",
        "enforce_invertibility",
        "simple_differencing",
        "measurement_error",
        "mle_regression",
        "hamilton_representation",
        "concentrate_scale",
    ]

    def __init__(
        self,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        start_params=None,
        method="lbfgs",
        maxiter=50,
        suppress_warnings=False,
        out_of_sample_size=0,
        scoring="mse",
        scoring_args=None,
        trend=None,
        with_intercept=True,
        time_varying_regression=False,
        enforce_stationarity=True,
        enforce_invertibility=True,
        simple_differencing=False,
        measurement_error=False,
        mle_regression=True,
        hamilton_representation=False,
        concentrate_scale=False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.start_params = start_params
        self.method = method
        self.maxiter = maxiter
        self.suppress_warnings = suppress_warnings
        self.out_of_sample_size = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = scoring_args
        self.trend = trend
        self.with_intercept = with_intercept
        for key in self.SARIMAX_KWARGS_KEYS:
            setattr(self, key, eval(key))

        super().__init__()

    def _instantiate_model(self):
        # import inside method to avoid hard dependency
        from pmdarima.arima.arima import ARIMA as _ARIMA

        sarimax_kwargs = {key: getattr(self, key) for key in self.SARIMAX_KWARGS_KEYS}

        return _ARIMA(
            order=self.order,
            seasonal_order=self.seasonal_order,
            start_params=self.start_params,
            method=self.method,
            maxiter=self.maxiter,
            suppress_warnings=self.suppress_warnings,
            out_of_sample_size=self.out_of_sample_size,
            scoring=self.scoring,
            scoring_args=self.scoring_args,
            trend=self.trend,
            with_intercept=self.with_intercept,
            **sarimax_kwargs,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params1 = {"maxiter": 3}
        params2 = {
            "order": (1, 1, 0),
            "seasonal_order": (1, 0, 0, 2),
            "maxiter": 3,
        }
        return [params1, params2]
