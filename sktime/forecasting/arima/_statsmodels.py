# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface to ARIMA from statsmodels package."""

__all__ = ["StatsModelsARIMA"]
__author__ = ["arnaujc91"]

from collections.abc import Iterable
from typing import Optional, Union

import numpy as np
import pandas as pd

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class StatsModelsARIMA(_StatsModelsAdapter):
    """(S)ARIMA(X) forecaster, from statsmodels, tsa.arima module.

    Direct interface for ``statsmodels.tsa.arima.model.ARIMA``.

    Users should note that statsmodels contains two separate implementations of
    (S)ARIMA(X), the ARIMA and the SARIMAX class, in different modules:
    ``tsa.arima.model.ARIMA`` and ``tsa.statespace.SARIMAX``.

    These are implementations of the same underlying model, (S)ARIMA(X),
    but with different
    fitting strategies, fitted parameters, and slightly differring behaviour.
    Users should refer to the statsmodels documentation for further details:
    https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_faq.html

    Parameters
    ----------
    order : tuple, optional
        The (p,d,q) order of the model for the autoregressive, differences, and
        moving average components. d is always an integer, while p and q may
        either be integers or lists of integers.
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. Default
        is (0, 0, 0, 0). D and s are always integers, while P and Q
        may either be integers or lists of positive integers.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend. Can be specified as a
        string where 'c' indicates a constant term, 't' indicates a
        linear trend in time, and 'ct' includes both. Can also be specified as
        an iterable defining a polynomial, as in ``numpy.poly1d``, where
        ``[1,1,0,1]`` would denote :math:`a + bt + ct^3`. Default is 'c' for
        models without integration, and no trend for models with integration.
        Note that all trend terms are included in the model as exogenous
        regressors, which differs from how trends are included in ``SARIMAX``
        models.  See the Notes section for a precise definition of the
        treatment of trend terms.
    enforce_stationarity : bool, optional
        Whether or not to require the autoregressive parameters to correspond
        to a stationarity process.
    enforce_invertibility : bool, optional
        Whether or not to require the moving average parameters to correspond
        to an invertible process.
    concentrate_scale : bool, optional
        Whether or not to concentrate the scale (variance of the error term)
        out of the likelihood. This reduces the number of parameters by one.
        This is only applicable when considering estimation by numerical
        maximum likelihood.
    trend_offset : int, optional
        The offset at which to start time trend values. Default is 1, so that
        if ``trend='t'`` the trend is equal to 1, 2, ..., nobs. Typically is only
        set when the model created by extending a previous dataset.
    dates : array_like of datetime, optional
        If no index is given by ``endog`` or ``exog``, an array-like object of
        datetime objects can be provided.
    freq : str, optional
        If no index is given by ``endog`` or ``exog``, the frequency of the
        time-series may be specified here as a Pandas offset or offset string.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        If None, the default is given by Model.start_params.
    transformed : bool, optional
        Whether or not ``start_params`` is already transformed. Default is
        True.
    includes_fixed : bool, optional
        If parameters were previously fixed with the ``fix_params`` method,
        this argument describes whether or not ``start_params`` also includes
        the fixed parameters, in addition to the free parameters. Default
        is False.
    method : str, optional
        The method used for estimating the parameters of the model. Valid
        options include 'statespace', 'innovations_mle', 'hannan_rissanen',
        'burg', 'innovations', and 'yule_walker'. Not all options are
        available for every specification (for example 'yule_walker' can
        only be used with AR(p) models).
    method_kwargs : dict, optional
        Arguments to pass to the fit function for the parameter estimator
        described by the ``method`` argument.
    gls : bool, optional
        Whether or not to use generalized least squares (GLS) to estimate
        regression effects. The default is False if ``method='statespace'``
        and is True otherwise.
    gls_kwargs : dict, optional
        Arguments to pass to the GLS estimation fit method. Only applicable
        if GLS estimation is used (see ``gls`` argument for details).
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

        Default is 'opg' unless memory conservation is used to avoid
        computing the loglikelihood values for each observation, in which
        case the default is 'oim'.
    cov_kwds : dict or None, optional
        A dictionary of arguments affecting covariance matrix computation.

        **opg, oim, approx, robust, robust_approx**

        - 'approx_complex_step' : bool, optional - If True, numerical
          approximations are computed using complex-step methods. If False,
          numerical approximations are computed using finite difference
          methods. Default is True.
        - 'approx_centered' : bool, optional - If True, numerical
          approximations computed using finite difference methods use a
          centered approximation. Default is False.
    return_params : bool, optional
        Whether or not to return only the array of maximizing parameters.
        Default is False.
    low_memory : bool, optional
        If set to True, techniques are applied to substantially reduce
        memory usage. If used, some features of the results object will
        not be available (including smoothed results and in-sample
        prediction), although out-of-sample forecasting is possible.
        Default is False.

    See Also
    --------
    ARIMA
    SARIMAX
    AutoARIMA
    StatsForecastAutoARIMA

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.arima import StatsModelsARIMA
    >>> y = load_airline()
    >>> forecaster = StatsModelsARIMA(order=(0, 0, 12))  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["chadfulton", "bashtage", "jbrockmendel", "arnaujc91"],
        # chadfulton, bashtage, jbrockmendel for statsmodels implementation
        "maintainers": ["arnaujc91"],
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["statsmodels"],
    }

    def __init__(
        self,
        order: tuple[int, int, int] = (0, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[Union[str, Iterable]] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        concentrate_scale: bool = False,
        trend_offset: int = 1,
        dates: Optional[np.ndarray] = None,
        freq: Optional[str] = None,
        missing: Optional[str] = None,
        validate_specification: bool = True,
        start_params: Optional[np.ndarray] = None,
        transformed: bool = True,
        includes_fixed: bool = False,
        method: Optional[str] = None,
        method_kwargs: Optional[dict] = None,
        gls: bool = False,
        gls_kwargs: Optional[dict] = None,
        cov_type: str = "opg",
        cov_kwds: Optional[dict] = None,
        return_params: bool = False,
        low_memory: bool = False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.concentrate_scale = concentrate_scale
        self.trend_offset = trend_offset
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.validate_specification = validate_specification

        # Fit params
        self.start_params = start_params
        self.transformed = transformed
        self.includes_fixed = includes_fixed
        self.method = method
        self.method_kwargs = method_kwargs
        self.gls = gls
        self.gls_kwargs = gls_kwargs
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.return_params = return_params
        self.low_memory = low_memory

        super().__init__()

    def _fit_forecaster(self, y, X=None):
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA

        self._forecaster = _ARIMA(
            endog=y,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            concentrate_scale=self.concentrate_scale,
            trend_offset=self.trend_offset,
            dates=self.dates,
            freq=self.freq,
            missing=self.missing,
            validate_specification=self.validate_specification,
        )
        self._fitted_forecaster = self._forecaster.fit(
            start_params=self.start_params,
            transformed=self.transformed,
            includes_fixed=self.includes_fixed,
            method=self.method,
            method_kwargs=self.method_kwargs,
            gls=self.gls,
            gls_kwargs=self.gls_kwargs,
            cov_type=self.cov_type,
            cov_kwds=self.cov_kwds,
            return_params=self.return_params,
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
        params : list of dict, default = []
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return [
            {
                "order": (0, 1, 2),
                "trend": "n",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
                "concentrate_scale": True,
                "method": "statespace",
            },
            {
                "order": (1, 1, 2),
                "trend": "t",
                "enforce_stationarity": False,
                "enforce_invertibility": False,
                "method": "statespace",
            },
            {
                "order": (0, 0, 1),
                "trend": "ct",
                "seasonal_order": (1, 0, 1, 2),
                "cov_type": "opg",
                "gls": True,
                "method": "statespace",
            },
            {"cov_type": "robust", "gls": True, "method": "burg"},
        ]
