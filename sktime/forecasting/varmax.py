"""Vector Autoregressive Moving Average with eXogenous regressors model (VARMAX)."""

__all__ = ["VARMAX"]
__author__ = ["KatieBuc"]

import warnings

import pandas as pd

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class VARMAX(_StatsModelsAdapter):
    r"""VARMAX forecasting model from statsmodels.

    Direct interface to ``VARMAX`` from ``statsmodels.tsa.statespace.varmax``.

    Vector Autoregressive Moving Average with eXogenous regressors model (VARMAX)

    Parameters
    ----------
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, ``[1,1,0,1]`` denotes
        :math:`a + bt + ct^3`. Default is a constant trend component.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations ``endog`` were
        measured with error. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    trend_offset : int, optional
        The offset at which to start time trend values. Default is 1, so that
        if ``trend='t'`` the trend is equal to 1, 2, ..., n_obs. Typically is only
        set when the model created by extending a previous dataset.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization. If
        None, the default is given by Model.start_params.
    transformed : bool, optional
        Whether or not start_params is already transformed. Default is True.
    includes_fixed : bool, optional
        If parameters were previously fixed with the fix_params method, this
        argument describes whether or not start_params also includes the
        fixed parameters, in addition to the free parameters. Default is False.
    cov_type : str, optional
        The ``cov_type`` keyword governs the method for calculating the
        covariance matrix of parameter estimates. Can be one of:
         - 'opg' for the outer product of gradient estimator
         - 'oim' for the observed information matrix estimator, calculated
            using the method of Harvey (1989)
         - 'approx' for the observed information matrix estimator,
            calculated using a numerical approximation of the Hessian matrix.
         - 'robust' for an approximate (quasi-maximum likelihood) covariance
            matrix that may be valid even in the presence of some
            misspecifications. Intermediate calculations use the 'oim'
            method.
         - 'robust_approx' is the same as 'robust' except that the
            intermediate calculations use the 'approx' method.
         - 'none' for no covariance matrix calculation.
        Default is 'opg' unless memory conservation is used to avoid computing the
        loglikelihood values for each observation, in which case the default is
        'approx'.
    cov_kwds : dict or None, optional
        A dictionary of arguments affecting covariance matrix computation.
        opg, oim, approx, robust, robust_approx
         - 'approx_complex_step' : bool, optional - If True, numerical
            approximations are computed using complex-step methods. If False,
            numerical approximations are computed using finite difference
            methods. Default is True.
         - 'approx_centered' : bool, optional - If True, numerical
            approximations computed using finite difference methods use a
            centered approximation. Default is False.
    method : str, optional
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
        solvers. See the notes section below (or scipy.optimize) for the
        available arguments and for the list of explicit arguments that the
        basin-hopping solver supports.
    maxiter : int, optional
        The maximum number of iterations to perform.
    full_output : bool, optional
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    disp : bool, optional
        Set to True to print convergence messages.
    callback : callable callback(xk), optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    return_params : bool, optional
        Whether or not to return only the array of maximizing parameters.
        Default is False.
    optim_score : {'harvey', 'approx'} or None, optional
        The method by which the score vector is calculated. 'harvey' uses
        the method from Harvey (1989), 'approx' uses either finite
        difference or complex step differentiation depending upon the
        value of ``optim_complex_step``, and None uses the built-in gradient
        approximation of the optimizer. Default is None. This keyword is
        only relevant if the optimization method uses the score.
    optim_complex_step : bool, optional
        Whether or not to use complex step differentiation when
        approximating the score; if False, finite difference approximation
        is used. Default is True. This keyword is only relevant if
        ``optim_score`` is set to 'harvey' or 'approx'.
    optim_hessian : {'opg','oim','approx'}, optional
        The method by which the Hessian is numerically approximated. 'opg'
        uses outer product of gradients, 'oim' uses the information
        matrix formula from Harvey (1989), and 'approx' uses numerical
        approximation. This keyword is only relevant if the
        optimization method uses the Hessian matrix.
    low_memory : bool, optional
        If set to True, techniques are applied to substantially reduce
        memory usage. If used, some features of the results object will
        not be available (including smoothed results and in-sample
        prediction), although out-of-sample forecasting is possible.
        Default is False.
    dynamic : bool, int, str, or datetime, optional
        Integer offset relative to ``start`` at which to begin dynamic
        prediction. Can also be an absolute date string to parse or a
        datetime type (these are not interpreted as offsets).
        Prior to this observation, true endogenous values will be used for
        prediction; starting with this observation and continuing through
        the end of prediction, forecasted endogenous values will be used
        instead.
    information_set : str, optional
        The information set to condition each prediction on. Default is
        "predicted", which computes predictions of period t values
        conditional on observed data through period t-1; these are
        one-step-ahead predictions, and correspond with the typical
        ``fittedvalues`` results attribute. Alternatives are "filtered",
        which computes predictions of period t values conditional on
        observed data through period t, and "smoothed", which computes
        predictions of period t values conditional on the entire dataset
        (including also future observations t+1, t+2, ...).
    signal_only : bool, optional
        Whether to compute predictions of only the "signal" component of
        the observation equation. Default is False. For example, the
        observation equation of a time-invariant model is
        :math:`y_t = d + Z \alpha_t + \varepsilon_t`, and the "signal"
        component is then :math:`Z \alpha_t`. If this argument is set to
        True, then predictions of the "signal" :math:`Z \alpha_t` will be
        returned. Otherwise, the default is for predictions of :math:`y_t`
        to be returned.
    suppress_warnings : bool, optional
        Many warnings might be thrown inside of statsmodels. If
        ``suppress_warnings`` is True, all of these warnings will be squelched.
        Default is False.

    Notes
    -----
    Generically, the VARMAX model is specified (see for example chapter 18 of
    [1]_):
    .. math::
        y_t = A(t) + A_1 y_{t-1} + \dots + A_p y_{t-p} + B x_t + \epsilon_t +
        M_1 \epsilon_{t-1} + \dots M_q \epsilon_{t-q}
    where :math:`\epsilon_t \sim N(0, \Omega)`, and where :math:`y_t` is a
    ``k_endog x 1`` vector. Additionally, this model allows considering the case
    where the variables are measured with error.
    Note that in the full VARMA(p,q) case there is a fundamental identification
    problem in that the coefficient matrices :math:`\{A_i, M_j\}` are not
    generally unique, meaning that for a given time series process there may
    be multiple sets of matrices that equivalently represent it. See Chapter 12
    of [1]_ for more information. Although this class can be used to estimate
    VARMA(p,q) models, a warning is issued to remind users that no steps have
    been taken to ensure identification in this case.

    References
    ----------
    .. [1] Lütkepohl, Helmut. 2007.
       New Introduction to Multiple Time Series Analysis.
       Berlin: Springer.

    Examples
    --------
    >>> from sktime.forecasting.varmax import VARMAX
    >>> from sktime.datasets import load_macroeconomic
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_macroeconomic()  # doctest: +SKIP
    >>> forecaster = VARMAX(suppress_warnings=True)  # doctest: +SKIP
    >>> forecaster.fit(y[['realgdp', 'unemp']])  # doctest: +SKIP
    VARMAX(...)
    >>> y_pred = forecaster.predict(fh=[1,4,12])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ChadFulton", "bashtage", "KatieBuc"],
        # ChadFulton and bashtage for statsnodels VARMAX
        # "python_dependencies": "statsmodels" - inherited from _StatsModelsAdapter
        # estimator type
        # --------------
        "scitype:y": "multivariate",
        "ignores-exogeneous-X": False,
        "handles-missing-data": False,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
    }

    def __init__(
        self,
        order=(1, 0),
        trend="c",
        error_cov_type="unstructured",
        measurement_error=False,
        enforce_stationarity=True,
        enforce_invertibility=True,
        trend_offset=1,
        start_params=None,
        transformed=True,
        includes_fixed=False,
        cov_type=None,
        cov_kwds=None,
        method="lbfgs",
        maxiter=50,
        full_output=1,
        disp=False,
        callback=None,
        return_params=False,
        optim_score=None,
        optim_complex_step=None,
        optim_hessian=None,
        flags=None,
        low_memory=False,
        dynamic=False,
        information_set="predicted",
        signal_only=False,
        suppress_warnings=False,
    ):
        # Model parameters
        self.order = order
        self.trend = trend
        self.error_cov_type = error_cov_type
        self.measurement_error = measurement_error
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.trend_offset = trend_offset
        self.start_params = start_params
        self.transformed = transformed
        self.includes_fixed = includes_fixed
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.method = method
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.return_params = return_params
        self.optim_score = optim_score
        self.optim_complex_step = optim_complex_step
        self.optim_hessian = optim_hessian
        self.flags = flags
        self.low_memory = low_memory
        self.dynamic = dynamic
        self.information_set = information_set
        self.signal_only = signal_only
        self.suppress_warnings = suppress_warnings

        super().__init__()

    def _fit_forecaster(self, y, X=None):
        """Fit forecaster to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : array_like
            The observed time-series process :math:`y`, shaped n_obs x k_endog.
        X : array_like, optional (default=None)
            Array of exogenous regressors, shaped n_obs x k.

        Returns
        -------
        self : reference to self
        """
        if self.suppress_warnings:
            warnings.filterwarnings("ignore")

        from statsmodels.tsa.statespace.varmax import VARMAX as _VARMAX

        self._forecaster = _VARMAX(
            endog=y,
            exog=X,
            order=self.order,
            trend=self.trend,
            error_cov_type=self.error_cov_type,
            measurement_error=self.measurement_error,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            trend_offset=self.trend_offset,
        )
        self._fitted_forecaster = self._forecaster.fit(
            start_params=self.start_params,
            transformed=self.transformed,
            includes_fixed=self.includes_fixed,
            cov_type=self.cov_type,
            cov_kwds=self.cov_kwds,
            method=self.method,
            maxiter=self.maxiter,
            full_output=self.full_output,
            disp=self.disp,
            callback=self.callback,
            return_params=self.return_params,
            optim_score=self.optim_score,
            optim_complex_step=self.optim_complex_step,
            optim_hessian=self.optim_hessian,
            flags=self.flags,
            low_memory=self.low_memory,
        )
        return self

    # defining `_predict`, instead of inheriting from `_StatsModelsAdapter`,
    # for two reasons:
    # 1. to pass in `dynamic`, `information_set` and `signal_only`
    # 2. to deal with statsmodel integer indexing issue
    def _predict(self, fh, X):
        """Wrap Statmodel's VARMAX forecast method.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasters horizon with the steps ahead to to predict.
            Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.

        Returns
        -------
        y_pred : np.ndarray
            Returns series of predicted values.
        """
        abs_idx = fh.to_absolute_int(self._y.index[0], self.cutoff)
        start, end = abs_idx[[0, -1]]
        full_range = pd.RangeIndex(start=start, stop=end + 1)

        y_pred = self._fitted_forecaster.predict(
            start=start,
            end=end,
            dynamic=self.dynamic,
            information_set=self.information_set,
            signal_only=self.signal_only,
            exog=X,
        )

        y_pred.index = full_range
        y_pred = y_pred.loc[abs_idx.to_pandas()]
        y_pred.index = fh.to_absolute_index(self.cutoff)

        return y_pred

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
        params = [
            {"order": (1, 0)},
            {"order": (0, 1)},
            {"order": (1, 1)},
        ]

        return params
