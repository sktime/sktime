# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements SARIMAX."""

__all__ = ["SARIMAX"]
__author__ = ["TNTran92", "yarnabrina"]

import pandas as pd

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class SARIMAX(_StatsModelsAdapter):
    """(S)ARIMA(X) forecaster, from statsmodels, tsa.statespace module.

    Direct interface for ``statsmodels.tsa.statespace.SARIMAX``.

    Users should note that statsmodels contains two separate implementations of
    (S)ARIMA(X), the ARIMA and the SARIMAX class, in different modules:
    ``tsa.arima.model.ARIMA`` and ``tsa.statespace.SARIMAX``.

    These are implementations of the same underlying model, (S)ARIMA(X),
    but with different
    fitting strategies, fitted parameters, and slightly differing behaviour.
    Users should refer to the statsmodels documentation for further details:
    https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_faq.html

    Parameters
    ----------
    order : iterable or iterable of iterables, optional, default=(1,0,0)
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters. ``d`` must be an integer
        indicating the integration order of the process, while
        ``p`` and ``q`` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. Default is
        an AR(1) model: (1,0,0).
    seasonal_order : iterable, optional, default=(0,0,0,0)
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity.
        ``D`` must be an integer indicating the integration order of the process,
        while ``P`` and ``Q`` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. ``s`` is an
        integer giving the periodicity (number of periods in season), often it
        is 4 for quarterly data or 12 for monthly data. Default is no seasonal
        effect.
    trend : str{'n','c','t','ct'} or iterable, optional, default="c"
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, ``[1,1,0,1]`` denotes
        :math:`a + bt + ct^3`. Default is to not include a trend component.
    measurement_error : bool, optional, default=False
        Whether or not to assume the endogenous observations ``endog`` were
        measured with error.
    time_varying_regression : bool, optional, default=False
        Used when an explanatory variables, ``exog``, are provided
        to select whether or not coefficients on the exogenous regressors are
        allowed to vary over time.
    mle_regression : bool, optional, default=True
        Whether or not to use estimate the regression coefficients for the
        exogenous variables as part of maximum likelihood estimation or through
        the Kalman filter (i.e. recursive least squares). If
        ``time_varying_regression`` is True, this must be set to False.
    simple_differencing : bool, optional, default=False
        Whether or not to use partially conditional maximum likelihood
        estimation. If True, differencing is performed prior to estimation,
        which discards the first :math:`s D + d` initial rows but results in a
        smaller state-space formulation. See the Notes section for important
        details about interpreting results when this option is used. If False,
        the full SARIMAX model is put in state-space form so that all
        datapoints can be used in estimation.
    enforce_stationarity : bool, optional, default=True
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model.
    enforce_invertibility : bool, optional, default=True
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model.
    hamilton_representation : bool, optional, default=False
        Whether or not to use the Hamilton representation of an ARMA process
        (if True) or the Harvey representation (if False).
    concentrate_scale : bool, optional, default=False
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters estimated
        by maximum likelihood by one, but standard errors will then not
        be available for the scale parameter.
    trend_offset : int, optional, default=1
        The offset at which to start time trend values. Default is 1, so that
        if ``trend='t'`` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    use_exact_diffuse : bool, optional, default=False
        Whether or not to use exact diffuse initialization for non-stationary
        states. Default is False (in which case approximate diffuse
        initialization is used).
    disp : bool, optional, default=False
        Set to True to print convergence messages.
    random_state : int, RandomState instance or None, optional, default=None
        default=None - If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by np.random.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization. If None, the
        default is given by SARIMAX.start_params.
    transformed : bool, optional
        Whether or not ``start_params`` is already transformed. Default is True.
    includes_fixed : bool, optional
        If parameters were previously fixed with the ``fix_params`` method, this
        argument
        describes whether or not ``start_params`` also includes the fixed parameters, in
        addition to the free parameters. Default is False.
    cov_type : str, optional
        The ``cov_type`` keyword governs the method for calculating the covariance
        matrix
        of parameter estimates. Can be one of:

        - 'opg' for the outer product of gradient estimator
        - 'oim' for the observed information matrix estimator, calculated
            using the method of Harvey (1989)
        - 'approx' for the observed information matrix estimator, calculated using a
            numerical approximation of the Hessian matrix.
        - 'robust' for an approximate (quasi-maximum likelihood) covariance matrix that
            may be valid even in the presence of some misspecifications. Intermediate
            calculations use the 'oim' method.
        - 'robust_approx' is the same as 'robust' except that the intermediate
            calculations use the 'approx' method.
        - 'none' for no covariance matrix calculation.

        Default is 'opg' unless memory conservation is used to avoid
        computing the loglikelihood values for each observation, in which
        case the default is 'approx'.
    cov_kwds : dict or None, optional
        A dictionary of arguments affecting covariance matrix computation.

        **opg, oim, approx, robust, robust_approx**

        - 'approx_complex_step' : bool, optional - If True, numerical
            approximations are computed using complex-step methods. If False, numerical
            approximations are computed using finite difference methods. Default is
            True.
        - 'approx_centered' : bool, optional - If True, numerical
            approximations computed using finite difference methods use a centered
            approximation. Default is False.
    method : str, optional
        The ``method`` determines which solver from ``scipy.optimize`` is used, and it
        can
        be chosen from among the following strings:

        - 'newton' for Newton-Raphson
        - 'nm' for Nelder-Mead
        - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        - 'lbfgs' for limited-memory BFGS with optional box constraints
        - 'powell' for modified Powell's method
        - 'cg' for conjugate gradient
        - 'ncg' for Newton-conjugate gradient
        - 'basinhopping' for global basin-hopping solver

        The explicit arguments in ``fit`` are passed to the solver, with the exception
        of
        the basin-hopping solver. Each solver has several optional arguments that are
        not the same across solvers. See the notes section below (or scipy.optimize) for
        the available arguments and for the list of explicit arguments that the
        basin-hopping solver supports.
    maxiter : int, optional
        The maximum number of iterations to perform.
    full_output : bool, optional
        Set to True to have all available output in the Results object's mle_retvals
        attribute. The output is dependent on the solver. See LikelihoodModelResults
        notes section for more information.
    callback : callable callback(xk), optional
        Called after each iteration, as callback(xk), where xk is the current parameter
        vector.
    return_params : bool, optional
        Whether or not to return only the array of maximizing parameters. Default is
        False.
    optim_score : {'harvey', 'approx'} or None, optional
        The method by which the score vector is calculated. 'harvey' uses the method
        from Harvey (1989), 'approx' uses either finite difference or complex step
        differentiation depending upon the value of ``optim_complex_step``, and None
        uses
        the built-in gradient approximation of the optimizer. Default is None. This
        keyword is only relevant if the optimization method uses the score.
    optim_complex_step : bool, optional
        Whether or not to use complex step differentiation when approximating the score;
        if False, finite difference approximation is used. Default is True. This keyword
        is only relevant if ``optim_score`` is set to 'harvey' or 'approx'.
    optim_hessian : {'opg','oim','approx'}, optional
        The method by which the Hessian is numerically approximated. 'opg' uses outer
        product of gradients, 'oim' uses the information matrix formula from Harvey
        (1989), and 'approx' uses numerical approximation. This keyword is only relevant
        if the optimization method uses the Hessian matrix.
    low_memory : bool, optional
        If set to True, techniques are applied to substantially reduce memory usage. If
        used, some features of the results object will not be available (including
        smoothed results and in-sample prediction), although out-of-sample forecasting
        is possible. Default is False.

    See Also
    --------
    ARIMA
    AutoARIMA
    StatsForecastAutoARIMA

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles
    and practice. OTexts, 2014.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.sarimax import SARIMAX
    >>> y = load_airline()
    >>> forecaster = SARIMAX(
    ...     order=(1, 0, 0), trend="t", seasonal_order=(1, 0, 0, 6))  # doctest: +SKIP
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    SARIMAX(...)
    >>> y_pred = forecaster.predict(fh=y.index)  # doctest: +SKIP
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "chadfulton",  # for statsmodels
            "bashtage",  # for statsmodels
            "jbrockmendel",  # for statsmodels
            "jackzyliu",  # for statsmodels
            "TNTran92",
            "yarnabrina",
        ],
        "maintainers": ["TNTran92", "yarnabrina"],
        # "python_dependencnies": "statsmodels" - inherited from _StatsModelsAdapter
        # estimator type
        # --------------
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        measurement_error=False,
        time_varying_regression=False,
        mle_regression=True,
        simple_differencing=False,
        enforce_stationarity=True,
        enforce_invertibility=True,
        hamilton_representation=False,
        concentrate_scale=False,
        trend_offset=1,
        use_exact_diffuse=False,
        dates=None,
        freq=None,
        missing="none",
        validate_specification=True,
        disp=False,
        random_state=None,
        start_params=None,
        transformed=True,
        includes_fixed=False,
        cov_type=None,
        cov_kwds=None,
        method="lbfgs",
        maxiter=50,
        full_output=1,
        callback=None,
        return_params=False,
        optim_score=None,
        optim_complex_step=None,
        optim_hessian=None,
        low_memory=False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.measurement_error = measurement_error
        self.time_varying_regression = time_varying_regression
        self.mle_regression = mle_regression
        self.simple_differencing = simple_differencing
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.hamilton_representation = hamilton_representation
        self.concentrate_scale = concentrate_scale
        self.trend_offset = trend_offset
        self.use_exact_diffuse = use_exact_diffuse
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.validate_specification = validate_specification

        # Fit params
        self.disp = disp
        self.start_params = start_params
        self.transformed = transformed
        self.includes_fixed = includes_fixed
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.method = method
        self.maxiter = maxiter
        self.full_output = full_output
        self.callback = callback
        self.return_params = return_params
        self.optim_score = optim_score
        self.optim_complex_step = optim_complex_step
        self.optim_hessian = optim_hessian
        self.low_memory = low_memory

        super().__init__(random_state=random_state)

    def _fit_forecaster(self, y, X=None):
        from statsmodels.tsa.api import SARIMAX as _SARIMAX

        self._forecaster = _SARIMAX(
            endog=y,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            measurement_error=self.measurement_error,
            time_varying_regression=self.time_varying_regression,
            mle_regression=self.mle_regression,
            simple_differencing=self.simple_differencing,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            hamilton_representation=self.hamilton_representation,
            concentrate_scale=self.concentrate_scale,
            trend_offset=self.trend_offset,
            use_exact_diffuse=self.use_exact_diffuse,
            dates=self.dates,
            freq=self.freq,
            missing=self.missing,
            validate_specification=self.validate_specification,
        )
        self._fitted_forecaster = self._forecaster.fit(
            disp=self.disp,
            start_params=self.start_params,
            transformed=self.transformed,
            includes_fixed=self.includes_fixed,
            cov_type=self.cov_type,
            cov_kwds=self.cov_kwds,
            method=self.method,
            maxiter=self.maxiter,
            full_output=self.full_output,
            callback=self.callback,
            return_params=self.return_params,
            optim_score=self.optim_score,
            optim_complex_step=self.optim_complex_step,
            optim_hessian=self.optim_hessian,
            low_memory=self.low_memory,
        )

    def summary(self):
        """Get a summary of the fitted forecaster.

        This is the same as the implementation in statsmodels:

        https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_structural_harvey_jaeger.html
        """
        return self._fitted_forecaster.summary()

    @staticmethod
    def _extract_conf_int(prediction_results, alpha) -> pd.DataFrame:
        """Construct confidence interval at specified ``alpha`` for each timestep.

        Parameters
        ----------
        prediction_results : PredictionResults
            results class, as returned by ``self._fitted_forecaster.get_prediction``
        alpha : float
            one minus nominal coverage

        Returns
        -------
        pd.DataFrame
            confidence intervals at each timestep

            The dataframe must have at least two columns ``lower`` and ``upper``, and
            the row indices must be integers relative to ``self.cutoff``. Order of
            columns do not matter, and row indices must be a superset of relative
            integer horizon of ``fh``.
        """
        conf_int = prediction_results.conf_int(alpha=alpha)
        conf_int.columns = ["lower", "upper"]

        return conf_int

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return [
            # this fails - seems like statsmodels error
            # {
            #     "order": (4, 1, 2),
            #     "trend": "ct",
            #     "time_varying_regression": True,
            #     "enforce_stationarity": False,
            #     "enforce_invertibility": False,
            #     "concentrate_scale": True,
            #     "use_exact_diffuse": True,
            #     "mle_regression": False,
            # },
            {
                "order": (2, 1, 2),
                "trend": "ct",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
            {
                "order": [1, 0, 1],
                "trend": [1, 1, 0, 1],
                # It does not work with measurement_error, not sure why.
                # "measurement_error": True,
                "seasonal_order": (1, 0, 1, 2),
                "hamilton_representation": True,
                "simple_differencing": True,
            },
            {"cov_type": "robust", "method": "bfgs", "maxiter": 5},
        ]
