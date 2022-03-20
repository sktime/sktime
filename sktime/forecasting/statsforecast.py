# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements AutoARIMA model from StatsForecast."""

__author__ = ["FedericoGarza"]
__all__ = ["AutoARIMA"]


from typing import Dict, Optional

from sktime.forecasting.base.adapters._statsforecast import _StatsForecastAdapter
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("statsforecast", severity="warning")


class AutoARIMA(_StatsForecastAdapter):
    """StatsForecast AutoARIMA estimator.

    Returns best ARIMA model according to either AIC, AICc or BIC value.
    The function conducts a search over possible model within
    the order constraints provided.

    Parameters
    ----------
    d: int optional (default None)
        Order of first-differencing.
        If missing, will choose a value based on `test`.
    D: int optional (default None)
        Order of seasonal-differencing.
        If missing, will choose a value based on `season_test`.
    max_p: int (default 5)
        Maximum value of p.
    max_q: int (default 5)
        Maximum value of q.
    max_P: int (default 2)
        Maximum value of P.
    max_Q: int (default 2)
        Maximum value of Q.
    max_order: int (default 5)
        Maximum value of p+q+P+Q if model selection is not stepwise.
    max_d: int (default 2)
        Maximum number of non-seasonal differences
    max_D: int (default 1)
        Maximum number of seasonal differences
    start_p: int (default 2)
        Starting value of p in stepwise procedure.
    start_q: int (default 2)
        Starting value of q in stepwise procedure.
    start_P: int (default 1)
        Starting value of P in stepwise procedure.
    start_Q: int (default 1)
        Starting value of Q in stepwise procedure.
    stationary: bool (default False)
        If True, restricts search to stationary models.
    seasonal: bool (default True)
        If False, restricts search to non-seasonal models.
    ic: str (default 'aicc')
        Information criterion to be used in model selection.
    stepwise: bool (default True)
        If True, will do stepwise selection (faster).
        Otherwise, it searches over all models.
        Non-stepwise selection can be very slow,
        especially for seasonal models.
    nmodels: int (default 94)
        Maximum number of models considered in the stepwise search.
    trace: bool (default False)
        If True, the list of ARIMA models considered will be reported.
    approximation: bool optional (default None)
        If True, estimation is via conditional sums of squares
        and the information criteria used for model
        selection are approximated.
        The final model is still computed using
        maximum likelihood estimation.
        Approximation should be used for long time series
        or a high seasonal period to avoid excessive computation times.
    method: str optional (default None)
        fitting method: maximum likelihood or minimize conditional
        sum-of-squares.
        The default (unless there are missing values)
        is to use conditional-sum-of-squares to find starting values,
        then maximum likelihood. Can be abbreviated.
    truncate: bool optional (default None)
        An integer value indicating how many observations
        to use in model selection.
        The last truncate values of the series are
        used to select a model when truncate is not None
        and approximation=True.
        All observations are used if either truncate=None
        or approximation=False.
    test: str (default 'kpss')
        Type of unit root test to use. See ndiffs for details.
    test_kwargs: str optional (default None)
        Additional arguments to be passed to the unit root test.
    seasonal_test: str (default 'seas')
        This determines which method is used to select the number
        of seasonal differences.
        The default method is to use a measure of seasonal
        strength computed from an STL decomposition.
        Other possibilities involve seasonal unit root tests.
    seasonal_test_kwargs: dict optional (default None)
        Additional arguments to be passed to the seasonal
        unit root test. See nsdiffs for details.
    allowdrift: bool (default True)
        If True, models with drift terms are considered.
    allowmean: bool (default True)
        If True, models with a non-zero mean are considered.
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
    num_cores: int (default 2)
        Allows the user to specify the amount of parallel processes to be used
        if parallel = True and stepwise = False.
        If None, then the number of logical cores is
        automatically detected and all available cores are used.
    period: int (default 1)
        Number of observations per unit of time.
        For example 24 for Hourly data.

    Notes
    -----
    * This implementation is a mirror of Hyndman's forecast::auto.arima.
    * This implementation is a wrapper of AutoARIMA from StatsForecast.

    References
    ----------
    [1] https://github.com/robjhyndman/forecast
    """

    def __init__(
        self,
        d: Optional[int] = None,
        D: Optional[int] = None,
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_order: int = 5,
        max_d: int = 2,
        max_D: int = 1,
        start_p: int = 2,
        start_q: int = 2,
        start_P: int = 1,
        start_Q: int = 1,
        stationary: bool = False,
        seasonal: bool = True,
        ic: str = "aicc",
        stepwise: bool = True,
        nmodels: int = 94,
        trace: bool = False,
        approximation: Optional[bool] = None,
        method: Optional[str] = None,
        truncate: Optional[bool] = None,
        test: str = "kpss",
        test_kwargs: Optional[str] = None,
        seasonal_test: str = "seas",
        seasonal_test_kwargs: Optional[Dict] = None,
        allowdrift: bool = True,
        allowmean: bool = True,
        blambda: Optional[float] = None,
        biasadj: bool = False,
        parallel: bool = False,
        num_cores: int = 2,
        period: int = 1,
    ):
        _check_soft_dependencies("statsforecast", severity="error", object=self)

        self.d = d
        self.D = D
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_order = max_order
        self.max_d = max_d
        self.max_D = max_D
        self.start_p = start_p
        self.start_q = start_q
        self.start_P = start_P
        self.start_Q = start_Q
        self.stationary = stationary
        self.seasonal = seasonal
        self.ic = ic
        self.stepwise = stepwise
        self.nmodels = nmodels
        self.trace = trace
        self.approximation = approximation
        self.method = method
        self.truncate = truncate
        self.test = test
        self.test_kwargs = test_kwargs
        self.seasonal_test = seasonal_test
        self.seasonal_test_kwargs = seasonal_test_kwargs
        self.allowdrift = allowdrift
        self.allowmean = allowmean
        self.blambda = blambda
        self.biasadj = biasadj
        self.parallel = parallel
        self.num_cores = num_cores
        self.period = period

        super(AutoARIMA, self).__init__()

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
            ic=self.ic,
            stepwise=self.stepwise,
            nmodels=self.nmodels,
            trace=self.trace,
            approximation=self.approximation,
            method=self.method,
            truncate=self.truncate,
            test=self.test,
            test_kwargs=self.test_kwargs,
            seasonal_test=self.seasonal_test,
            seasonal_test_kwargs=self.seasonal_test_kwargs,
            allowdrift=self.allowdrift,
            allowmean=self.allowmean,
            blambda=self.blambda,
            biasadj=self.biasadj,
            parallel=self.parallel,
            num_cores=self.num_cores,
            period=self.period,
        )

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

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
