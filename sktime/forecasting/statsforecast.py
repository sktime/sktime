# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from statsforecast by Nixtla."""

__author__ = ["AzulGarza", "yarnabrina"]

__all__ = [
    "StatsForecastAutoARIMA",
    "StatsForecastAutoCES",
    "StatsForecastAutoETS",
    "StatsForecastAutoTBATS",
    "StatsForecastAutoTheta",
    "StatsForecastMSTL",
]
from typing import Optional, Union

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base.adapters._generalised_statsforecast import (
    StatsForecastBackAdapter,
    _GeneralisedStatsForecastAdapter,
)
from sktime.utils.dependencies import _check_soft_dependencies


class StatsForecastAutoARIMA(_GeneralisedStatsForecastAdapter):
    """StatsForecast AutoARIMA estimator.

    Direct interface to ``statsforecast.models.AutoARIMA`` by Nixtla.

    This estimator directly interfaces ``AutoARIMA``,
    from ``statsforecast`` [2]_ by Nixtla.
    The ``statsforecast`` implementation is inspired
    by Hyndman's forecast::auto.arima [1]_.

    Returns best ARIMA model according to either AIC, AICc or BIC value.
    The function conducts a search over possible model within
    the order constraints provided.

    Parameters
    ----------
    start_p: int (default 2)
        Starting value of p in stepwise procedure.
    d: int optional (default None)
        Order of first-differencing.
        If missing, will choose a value based on ``test``.
    start_q: int (default 2)
        Starting value of q in stepwise procedure.
    max_p: int (default 5)
        Maximum value of p.
    max_d: int (default 2)
        Maximum number of non-seasonal differences
    max_q: int (default 5)
        Maximum value of q.
    start_P: int (default 1)
        Starting value of P in stepwise procedure.
    D: int optional (default None)
        Order of seasonal-differencing.
        If missing, will choose a value based on ``season_test``.
    start_Q: int (default 1)
        Starting value of Q in stepwise procedure.
    max_P: int (default 2)
        Maximum value of P.
    max_D: int (default 1)
        Maximum number of seasonal differences
    max_Q: int (default 2)
        Maximum value of Q.
    max_order: int (default 5)
        Maximum value of p+q+P+Q if model selection is not stepwise.
    sp: int (default 1)
        Number of observations per unit of time.
        For example 24 for Hourly data.
    seasonal: bool (default True)
        If False, restricts search to non-seasonal models.
    stationary: bool (default False)
        If True, restricts search to stationary models.
    information_criterion: str (default 'aicc')
        Information criterion to be used in model selection.
        It can be chosen from among the following strings:
        - 'aicc' for Akaike's information criterion corrected.
        - 'aic' for Akaike's information criterion.
        - 'bic' for bayesian information criterion.
    test: str (default 'kpss')
        Type of unit root test to use. See ndiffs for details.
        Only 'kpss' for the Kwiatkowski-Phillip-Schmidt-Shin test
        is allowed.
    seasonal_test: str (default 'seas')
        This determines which method is used to select the number
        of seasonal differences.
        The default method ('seas') is to use a measure of seasonal
        strength computed from an STL decomposition.
        Other possibilities involve seasonal unit root tests.
        Only 'seas' is allowed.
    stepwise: bool (default True)
        If True, will do stepwise selection (faster).
        Otherwise, it searches over all models.
        Non-stepwise selection can be very slow,
        especially for seasonal models.
    n_jobs: int (default 2)
        Allows the user to specify the amount of parallel processes to be used
        if parallel = True and stepwise = False.
        If None, then the number of logical cores is
        automatically detected and all available cores are used.
    trend: bool (default True)
        If True, models with drift terms are considered.
    method: str optional (default None)
        fitting method: maximum likelihood or minimize conditional
        sum-of-squares.
        The default (unless there are missing values)
        is to use conditional-sum-of-squares to find starting values,
        then maximum likelihood. Can be abbreviated.
        It can be chosen from among the following strings:

        - 'CSS-ML' for conditional sum-of-squares to find starting values and
          then maximum likelihood.
        - 'ML' for maximum likelihood.
        - 'CSS' for conditional sum-of-squares.

    offset_test_args: dict optional (default None)
        Additional arguments to be passed to the unit root test.
    seasonal_test_args: dict optional (default None)
        Additional arguments to be passed to the seasonal
        unit root test. See nsdiffs for details.
    trace: bool (default False)
        If True, the list of ARIMA models considered will be reported.
    n_fits: int (default 94)
        Maximum number of models considered in the stepwise search.
    with_intercept: bool (default True)
        If True, models with a non-zero mean are considered.
    approximation: bool optional (default None)
        If True, estimation is via conditional sums of squares
        and the information criteria used for model
        selection are approximated.
        The final model is still computed using
        maximum likelihood estimation.
        Approximation should be used for long time series
        or a high seasonal period to avoid excessive computation times.
    truncate: bool optional (default None)
        An integer value indicating how many observations
        to use in model selection.
        The last truncate values of the series are
        used to select a model when truncate is not None
        and approximation=True.
        All observations are used if either truncate=None
        or approximation=False.
    blambda: float optional (default None)
        Box-Cox transformation parameter.
        If lambda="auto", then a transformation is automatically
        selected using BoxCox.lambda.
        The transformation is ignored if None.
        Otherwise, data transformed before model is estimated.
    biasadj: bool (default False)
        Use adjusted back-transformed mean for Box-Cox transformations.
        If transformed data is used to produce forecasts and fitted values,
        a regular back transformation will result in median forecasts.
        If biasadj is True, an adjustment will be made to produce
        mean forecasts and fitted values.
    parallel: bool (default False)
        If True and stepwise = False, then the specification search
        is done in parallel.
        This can give a significant speedup on multicore machines.

    References
    ----------
    .. [1] https://github.com/robjhyndman/forecast
    .. [2] https://github.com/Nixtla/statsforecast

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
    >>> y = load_airline()
    >>> forecaster = StatsForecastAutoARIMA(  # doctest: +SKIP
    ...     sp=12, d=0, max_p=2, max_q=2
    ... )
    >>> forecaster.fit(y)  # doctest: +SKIP
    StatsForecastAutoARIMA(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["AzulGarza", "yarnabrina"],
        "maintainers": ["AzulGarza"],
        # "python_dependencies": "statsforecast"
        # inherited from _GeneralisedStatsForecastAdapter
        # estimator type
        # --------------
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["statsforecast>=1.0.0"],
    }

    def __init__(
        self,
        start_p: int = 2,
        d: Optional[int] = None,
        start_q: int = 2,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        start_P: int = 1,
        D: Optional[int] = None,
        start_Q: int = 1,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        max_order: int = 5,
        sp: int = 1,
        seasonal: bool = True,
        stationary: bool = False,
        information_criterion: str = "aicc",
        test: str = "kpss",
        seasonal_test: str = "seas",
        stepwise: bool = True,
        n_jobs: int = 2,
        trend: bool = True,
        method: Optional[str] = None,
        offset_test_args: Optional[str] = None,
        seasonal_test_args: Optional[dict] = None,
        trace: bool = False,
        n_fits: int = 94,
        with_intercept: bool = True,
        approximation: Optional[bool] = None,
        truncate: Optional[bool] = None,
        blambda: Optional[float] = None,
        biasadj: bool = False,
        parallel: bool = False,
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
        self.test = test
        self.seasonal_test = seasonal_test
        self.stepwise = stepwise
        self.n_jobs = n_jobs
        self.trend = trend
        self.method = method
        self.offset_test_args = offset_test_args
        self.seasonal_test_args = seasonal_test_args
        self.trace = trace
        self.n_fits = n_fits
        self.with_intercept = with_intercept
        self.approximation = approximation
        self.truncate = truncate
        self.blambda = blambda
        self.biasadj = biasadj
        self.parallel = parallel

        super().__init__()

    def _get_statsforecast_class(self):
        """Get the class of the statsforecast forecaster."""
        from statsforecast.models import AutoARIMA

        return AutoARIMA

    def _get_statsforecast_params(self):
        return {
            "d": self.d,
            "D": self.D,
            "max_p": self.max_p,
            "max_q": self.max_q,
            "max_P": self.max_P,
            "max_Q": self.max_Q,
            "max_order": self.max_order,
            "max_d": self.max_d,
            "max_D": self.max_D,
            "start_p": self.start_p,
            "start_q": self.start_q,
            "start_P": self.start_P,
            "start_Q": self.start_Q,
            "stationary": self.stationary,
            "seasonal": self.seasonal,
            "ic": self.information_criterion,
            "stepwise": self.stepwise,
            "nmodels": self.n_fits,
            "trace": self.trace,
            "approximation": self.approximation,
            "method": self.method,
            "truncate": self.truncate,
            "test": self.test,
            "test_kwargs": self.offset_test_args,
            "seasonal_test": self.seasonal_test,
            "seasonal_test_kwargs": self.seasonal_test_args,
            "allowdrift": self.trend,
            "allowmean": self.with_intercept,
            "blambda": self.blambda,
            "biasadj": self.biasadj,
            "parallel": self.parallel,
            "num_cores": self.n_jobs,
            "season_length": self.sp,
        }

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        params = [{}, {"approximation": True, "max_p": 4, "max_Q": 1}]

        return params


class StatsForecastAutoTheta(_GeneralisedStatsForecastAdapter):
    """Statsforecast AutoTheta estimator.

    Direct interface to ``statsforecast.models.AutoTheta`` by Nixtla.

    This estimator directly interfaces ``AutoTheta``,
    from ``statsforecast`` [1]_ by Nixtla.

    AutoTheta model automatically selects the best Theta (Standard Theta Model ("STM"),
    Optimized Theta Model ("OTM"), Dynamic Standard Theta Model ("DSTM"), Dynamic
    Optimized Theta Model ("DOTM")) model using mse.

    Parameters
    ----------
    season_length : int, optional, default=1
        number of observations per unit of time (e.g. 24 for hourly data), by default 1

    decomposition_type : str, optional, default="multipliciative"
        possible values: "additive", "multiplicative"
        type of seasonal decomposition, by default "multiplicative"

    model : Optional[str], optional
        controlling Theta Model, by default searches the best model

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotheta

    See Also
    --------
    ThetaForecaster
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "AzulGarza",
            "jmoralez",
            "yarnabrina",
            "arnaujc91",
            "luca-miniati",
        ],
        # AzulGarza, jmoralez for statsforecast theta
        # "maintainers": ["yarnabrina"],
        # "python_dependencies": "statsforecast"
        # inherited from _GeneralisedStatsForecastAdapter
        # estimator type
        # --------------
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["statsforecast>=1.3.0"],
    }

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        model: Optional[str] = None,
    ):
        self.season_length = season_length
        self.decomposition_type = decomposition_type
        self.model = model

        super().__init__()

    def _get_statsforecast_class(self):
        """Get the class of the statsforecast forecaster."""
        from statsforecast.models import AutoTheta

        return AutoTheta

    def _get_statsforecast_params(self):
        return {
            "season_length": self.season_length,
            "decomposition_type": self.decomposition_type,
            "model": self.model,
        }

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
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        params = [{}, {"season_length": 4}]

        return params


class StatsForecastAutoETS(_GeneralisedStatsForecastAdapter):
    """StatsForecast Automatic Exponential Smoothing model.

    Direct interface to ``statsforecast.models.AutoETS``,
    from ``statsforecast`` [1]_ by Nixtla.
    The ``statsforecast`` implementation is a mirror of Hyndman's forecast::ets [2]_.

    Automatically selects the best ETS (Error, Trend, Seasonality) model using an
    information criterion. Default is Akaike Information Criterion (AICc), while
    particular models are estimated using maximum likelihood. The state-space
    equations can be determined based on their $M$ multiplicative, $A$ additive, $Z$
    optimized or $N$ omitted components. The ``model`` string parameter defines the ETS
    equations: E in [$M, A, Z$], T in [$N, A, M, Z$], and S in [$N, A, M, Z$].

    For example when model='ANN' (additive error, no trend, and no seasonality), ETS
    will explore only a simple exponential smoothing.

    If the component is selected as 'Z', it operates as a placeholder to ask the
    AutoETS model to figure out the best parameter.

    Parameters
    ----------
    season_length : int
        Number of observations per unit of time. Ex: 24 Hourly data.
    model : str
        Controlling state-space-equations.
    damped : bool
        A parameter that 'dampens' the trend.
    phi : float, optional (default=None)
        Smoothing parameter for trend damping. Only used when ``damped=True``.

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoets
    .. [2] https://github.com/robjhyndman/forecast

    See Also
    --------
    AutoETS
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "AzulGarza",
            "jmoralez",
            "yarnabrina",
            "arnaujc91",
            "luca-miniati",
        ],
        # AzulGarza and jmoralez for statsforecast AutoETS
        # "maintainers": ["yarnabrina"],
        # "python_dependencies": "statsforecast"
        # inherited from _GeneralisedStatsForecastAdapter
        # estimator type
        # --------------
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["statsforecast>=1.3.2"],
    }

    def __init__(
        self,
        season_length: int = 1,
        model: str = "ZZZ",
        damped: Optional[bool] = None,
        phi: Optional[float] = None,
    ):
        self.season_length = season_length
        self.model = model
        self.damped = damped
        self.phi = phi

        super().__init__()

    def _get_statsforecast_class(self):
        """Create underlying forecaster instance."""
        from statsforecast.models import AutoETS

        return AutoETS

    def _get_statsforecast_params(self):
        return {
            "season_length": self.season_length,
            "model": self.model,
            "damped": self.damped,
            "phi": self.phi,
        }

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
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        params = [{}, {"season_length": 4, "model": "ZMZ"}]

        return params


class StatsForecastAutoCES(_GeneralisedStatsForecastAdapter):
    """StatsForecast Complex Exponential Smoothing model.

    Direct interface to ``statsforecast.models.AutoCES``,
    from ``statsforecast`` [1]_ by Nixtla.

    Automatically selects the best Complex Exponential Smoothing model using an
    information criterion. Default is Akaike Information Criterion (AICc), while
    particular models are estimated using maximum likelihood. The state-space equations
    can be determined based on their $S$ simple, $P$ partial, $Z$ optimized or $N$
    omitted components. The ``model`` string parameter defines the kind of CES model:
    $N$ for simple CES (without seasonality), $S$ for simple seasonality (lagged CES),
    $P$ for partial seasonality (without complex part), $F$ for full seasonality
    (lagged CES with real and complex seasonal parts).

    If the component is selected as 'Z', it operates as a placeholder to ask the
    AutoCES model to figure out the best parameter.

    Parameters
    ----------
    season_length : int
        Number of observations per unit of time. Ex: 24 Hourly data.
    model : str
        Controlling state-space-equations.

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoces
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "AzulGarza",
            "jmoralez",
            "MMenchero",
            "yarnabrina",
            "arnaujc91",
            "luca-miniati",
        ],
        # AzulGarza, jmoralez, MMenchero for statsforecast AutoCES
        # "maintainers": ["yarnabrina"],
        # "python_dependencies": "statsforecast"
        # inherited from _GeneralisedStatsForecastAdapter
        # estimator type
        # --------------
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["statsforecast>=1.1.0"],
    }

    def __init__(self, season_length: int = 1, model: str = "Z"):
        self.season_length = season_length
        self.model = model

        super().__init__()

    def _get_statsforecast_class(self):
        """Get the class of the statsforecast forecaster."""
        from statsforecast.models import AutoCES

        return AutoCES

    def _get_statsforecast_params(self):
        return {
            "season_length": self.season_length,
            "model": self.model,
        }

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
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        params = [{}, {"season_length": 4, "model": "Z"}]

        return params


class StatsForecastAutoTBATS(_GeneralisedStatsForecastAdapter):
    """StatsForecast TBATS model.

    Direct interface to ``statsforecast.models.AutoTBATS``,
    from ``statsforecast`` [1]_ by Nixtla.

    Automatically selects the best TBATS model from all feasible combinations of the
    parameters ``use_boxcox``, ``use_trend``, ``use_damped_trend``, and
    ``use_arma_errors``.
    Selection is made using the AIC.

    Default value for ``use_arma_errors`` is ``True`` since this enables the evaluation
    of
    models with and without ARMA errors.

    Parameters
    ----------
    seasonal_periods : int or list of int.
        Number of observations per unit of time. Ex: 24 Hourly data.
    use_boxcox : bool (default=None)
        Whether or not to use a Box-Cox transformation. By default tries both.
    use_trend : bool (default=None)
        Whether or not to use a trend component. By default tries both.
    use_damped_trend : bool (default=None)
        Whether or not to dampen the trend component. By default tries both.
    use_arma_errors : bool (default=True)
        Whether or not to use a ARMA errors.
        Default is True and this evaluates both models.
    bc_lower_bound : float (default=0.0)
        Lower bound for the Box-Cox transformation.
    bc_upper_bound : float (default=1.0)
        Upper bound for the Box-Cox transformation.

    See Also
    --------
    BATS
    TBATS

    References
    ----------
    .. [1] https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autotbats
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "MMenchero",
            "jmoralez",
            "yarnabrina",
            "arnaujc91",
            "luca-miniati",
        ],
        # MMenchero and jmoralez for statsforecast AutoTBATS
        # "maintainers": ["yarnabrina"],
        # "python_dependencies": "statsforecast"
        # inherited from _GeneralisedStatsForecastAdapter
        # estimator type
        # --------------
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["statsforecast>=1.7.2"],
    }

    def __init__(
        self,
        seasonal_periods: Union[int, list[int]],
        use_boxcox: Optional[bool] = None,
        use_trend: Optional[bool] = None,
        use_damped_trend: Optional[bool] = None,
        use_arma_errors: bool = True,
        bc_lower_bound: float = 0.0,
        bc_upper_bound: float = 1.0,
    ):
        self.seasonal_periods = seasonal_periods
        self.use_boxcox = use_boxcox
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.use_arma_errors = use_arma_errors
        self.bc_lower_bound = bc_lower_bound
        self.bc_upper_bound = bc_upper_bound

        super().__init__()

    def _get_statsforecast_class(self):
        """Create underlying forecaster instance."""
        from statsforecast.models import AutoTBATS

        return AutoTBATS

    def _get_statsforecast_params(self) -> dict:
        return {
            "season_length": self.seasonal_periods,
            "use_boxcox": self.use_boxcox,
            "use_trend": self.use_trend,
            "use_damped_trend": self.use_damped_trend,
            "use_arma_errors": self.use_arma_errors,
            "bc_lower_bound": self.bc_lower_bound,
            "bc_upper_bound": self.bc_upper_bound,
        }

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
        del parameter_set  # to avoid being detected as unused by `vulture` etc.

        params = [
            {
                "seasonal_periods": 3,
                "use_boxcox": True,
                "bc_lower_bound": 0.25,
                "bc_upper_bound": 0.75,
            },
            {
                "seasonal_periods": [3, 12],
                "use_boxcox": False,
                "use_trend": True,
                "use_damped_trend": True,
                "use_arma_errors": False,
            },
        ]

        return params


class StatsForecastMSTL(_GeneralisedStatsForecastAdapter):
    """StatsForecast Multiple Seasonal-Trend decomposition using LOESS model.

    Direct interface to ``statsforecast.models.MSTL``,
    from ``statsforecast`` [1]_ by Nixtla, with a back-adapter that allows
    to use ``sktime`` forecasters as trend forecasters.

    The MSTL (Multiple Seasonal-Trend decomposition using LOESS) decomposes the time
    series in multiple seasonalities using LOESS. Then forecasts the trend using
    a custom non-seasonal model (``trend_forecaster``) and each seasonality using a
    SeasonalNaive model. MSTL requires the input time series data to be univariate.

    Parameters
    ----------
    season_length : Union[int, List[int]]
        Number of observations per unit of time. For multiple seasonalities use a
        list.
    trend_forecaster : estimator, optional, default=StatsForecastAutoETS()
        Sktime estimator used to make univariate forecasts. Multivariate estimators are
        not supported.
    stl_kwargs : dict, optional
        Extra arguments to pass to [``statsmodels.tsa.seasonal.STL``]

        (https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html#statsmodels.tsa.seasonal.STL).
        The ``period`` and ``seasonal`` arguments are reserved.
    pred_int_kwargs : dict, optional
        Extra arguments to pass to [``statsforecast.utils.ConformalIntervals``].

    References
    ----------
    .. [1]
        https://nixtla.github.io/statsforecast/src/core/models.html#mstl

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.statsforecast import StatsForecastMSTL

    >>> y = load_airline()
    >>> model = StatsForecastMSTL(season_length=[3,12]) # doctest: +SKIP
    >>> fitted_model = model.fit(y=y) # doctest: +SKIP
    >>> y_pred = fitted_model.predict(fh=[1,2,3]) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "AzulGarza",
            "jmoralez",
            "luca-miniati",
            "yarnabrina",
            "arnaujc91",
        ],
        # AzulGarza and jmoralez for statsforecast MSTL
        "maintainers": "luca-miniati",
        # "python_dependencies": "statsforecast"
        # inherited from _GeneralisedStatsForecastAdapter
        # estimator type
        # --------------
        "ignores-exogeneous-X": True,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "python_dependencies": ["statsforecast>=1.2.0"],
    }

    def __init__(
        self,
        season_length: Union[int, list[int]],
        trend_forecaster=None,
        stl_kwargs: Optional[dict] = None,
        pred_int_kwargs: Optional[dict] = None,
    ):
        self.season_length = season_length
        self.trend_forecaster = trend_forecaster
        self.stl_kwargs = stl_kwargs
        self.pred_int_kwargs = pred_int_kwargs

        super().__init__()

        # adapter class sets probabilistic capability as true
        # because level is present in statsforecast signature
        # happens in _check_supports_pred_int method
        # manually overriding this temporarily
        self.set_tags(
            **{"capability:pred_int": False, "capability:pred_int:insample": False}
        )

        from sklearn.base import clone

        if trend_forecaster:
            self._trend_forecaster = clone(trend_forecaster)
        else:
            self._trend_forecaster = StatsForecastAutoETS(model="ZZN")

        # checks if trend_forecaster is already wrapped with
        # StatsForecastBackAdapter
        if not isinstance(self._trend_forecaster, StatsForecastBackAdapter):
            # if trend_forecaster is sktime forecaster
            if isinstance(self._trend_forecaster, BaseForecaster):
                self._trend_forecaster = StatsForecastBackAdapter(
                    self._trend_forecaster
                )
            else:
                raise TypeError(
                    "The provided forecaster is not compatible with MSTL. Please ensure"
                    " that the forecaster you pass into the model is a sktime "
                    "forecaster."
                )

        # check if prediction interval kwargs are passed
        if self.pred_int_kwargs:
            from statsforecast.utils import ConformalIntervals

            self._trend_forecaster.prediction_intervals = ConformalIntervals(
                **self.pred_int_kwargs
            )

    def _get_statsforecast_class(self):
        from statsforecast.models import MSTL

        return MSTL

    def _get_statsforecast_params(self):
        return {
            "season_length": self.season_length,
            "trend_forecaster": self._trend_forecaster,
        }

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
            Each dict are parameters to construct an "interesting" test instance,
            i.e., ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid
            test instance. ``create_test_instance`` uses the first (or only)
            dictionary in ``params``
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        try:
            _check_soft_dependencies("statsmodels")
            from sktime.forecasting.theta import ThetaForecaster

            params = [
                {
                    "season_length": [3, 12],
                    "trend_forecaster": ThetaForecaster(),
                },
                {
                    "season_length": 4,
                },
                {
                    "season_length": 4,
                    "pred_int_kwargs": {
                        "n_windows": 2,
                    },
                },
            ]
        except ModuleNotFoundError:
            from sktime.forecasting.naive import NaiveForecaster

            params = [
                {
                    "season_length": [3, 12],
                    "trend_forecaster": NaiveForecaster(),
                },
                {
                    "season_length": 4,
                },
            ]

        return params
