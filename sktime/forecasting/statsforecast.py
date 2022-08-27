# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements AutoARIMA model from statsforecast by Nixtla."""

__author__ = ["FedericoGarza"]
__all__ = ["StatsForecastAutoARIMA"]


from typing import Dict, Optional

from sktime.forecasting.base.adapters._statsforecast import _StatsForecastAdapter
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("statsforecast", severity="warning")


class StatsForecastAutoARIMA(_StatsForecastAdapter):
    """StatsForecast AutoARIMA estimator.

    This implementation is inspired by Hyndman's forecast::auto.arima [1]_
    and based on the Python implementation of statsforecast [2]_ by Nixtla.

    Returns best ARIMA model according to either AIC, AICc or BIC value.
    The function conducts a search over possible model within
    the order constraints provided.

    Parameters
    ----------
    start_p: int (default 2)
        Starting value of p in stepwise procedure.
    d: int optional (default None)
        Order of first-differencing.
        If missing, will choose a value based on `test`.
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
        If missing, will choose a value based on `season_test`.
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
        seasonal_test_args: Optional[Dict] = None,
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

        super(StatsForecastAutoARIMA, self).__init__()

    def _instantiate_model(self):
        # import inside method to avoid hard dependency
        from statsforecast.arima import AutoARIMA as _AutoARIMA

        return _AutoARIMA(
            d=self.d,
            D=self.D,
            max_p=self.max_p,
            max_q=self.max_q,
            max_P=self.max_P,
            max_Q=self.max_Q,
            max_order=self.max_order,
            max_d=self.max_d,
            max_D=self.max_D,
            start_p=self.start_p,
            start_q=self.start_q,
            start_P=self.start_P,
            start_Q=self.start_Q,
            stationary=self.stationary,
            seasonal=self.seasonal,
            ic=self.information_criterion,
            stepwise=self.stepwise,
            nmodels=self.n_fits,
            trace=self.trace,
            approximation=self.approximation,
            method=self.method,
            truncate=self.truncate,
            test=self.test,
            test_kwargs=self.offset_test_args,
            seasonal_test=self.seasonal_test,
            seasonal_test_kwargs=self.seasonal_test_args,
            allowdrift=self.trend,
            allowmean=self.with_intercept,
            blambda=self.blambda,
            biasadj=self.biasadj,
            parallel=self.parallel,
            num_cores=self.n_jobs,
            period=self.sp,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {"approximation": True, "max_p": 4, "max_Q": 1}
        return params
