# -*- coding: utf-8 -*-
__all__ = ["VARMAX"]
__author__ = ["KatieBuc"]

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX as _VARMAX

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class VARMAX(_StatsModelsAdapter):
    """
    Wrapper for statsmodels VARMAX model

    Vector Autoregressive Moving Average with eXogenous regressors model (VARMAX)

    Parameters
    ----------
    y : array_like
        The observed time-series process :math:`y`, , shaped n_obs x k_endog.
    X : array_like, optional
        Array of exogenous regressors, shaped n_obs x k.
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`. Default is a constant trend component.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    trend_offset : int, optional
        The offset at which to start time trend values. Default is 1, so that
        if `trend='t'` the trend is equal to 1, 2, ..., n_obs. Typically is only
        set when the model created by extending a previous dataset.
    **kwargs
        Keyword arguments may be used to provide default values for state space
        matrices or for Kalman filtering options. See `Representation`, and
        `KalmanFilter` for more details.
    Attributes
    ----------
    order : iterable
        The (p,q) order of the model for the number of AR and MA parameters to
        use.
    trend : str{'n','c','t','ct'} or iterable
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`.
    error_cov_type : {'diagonal', 'unstructured'}, optional
        The structure of the covariance matrix of the error term, where
        "unstructured" puts no restrictions on the matrix and "diagonal"
        requires it to be a diagonal matrix (uncorrelated errors). Default is
        "unstructured".
    measurement_error : bool, optional
        Whether or not to assume the endogenous observations `endog` were
        measured with error. Default is False.
    enforce_stationarity : bool, optional
        Whether or not to transform the AR parameters to enforce stationarity
        in the autoregressive component of the model. Default is True.
    enforce_invertibility : bool, optional
        Whether or not to transform the MA parameters to enforce invertibility
        in the moving average component of the model. Default is True.
    Notes
    -----
    Generically, the VARMAX model is specified (see for example chapter 18 of
    [1]_):
    .. math::
        y_t = A(t) + A_1 y_{t-1} + \dots + A_p y_{t-p} + B x_t + \epsilon_t +
        M_1 \epsilon_{t-1} + \dots M_q \epsilon_{t-q}
    where :math:`\epsilon_t \sim N(0, \Omega)`, and where :math:`y_t` is a
    `k_endog x 1` vector. Additionally, this model allows considering the case
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
    >>>from sktime.forecasting.varmax import VARMAX
    >>>from sktime.datasets import load_longley
    >>>_, y = load_longley()
    >>>train, test = y.iloc[:-3,], y.iloc[-3:,]
    >>>forecaster = VARMAX()
    >>>forecaster.fit(train[['GNPDEFL', 'GNP', 'UNEMP', 'POP']], X=train[['ARMED']])
    >>>y_pred = forecaster.predict(fh=[1,2,3], X=test[['ARMED']])
    """

    _tags = {
        "scitype:y": "multivariate",
        "ignores-exogeneous-X": False,
        "handles-missing-data": False,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:pred_int": False,
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
        disp=5,
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
        random_state=None,
    ):
        # Model parameters
        self.order=order
        self.trend=trend
        self.error_cov_type=error_cov_type
        self.measurement_error=measurement_error
        self.enforce_stationarity=enforce_stationarity
        self.enforce_invertibility=enforce_invertibility
        self.trend_offset=trend_offset
        self.start_params=start_params
        self.transformed=transformed
        self.includes_fixed=includes_fixed
        self.cov_type=cov_type
        self.cov_kwds=cov_kwds
        self.method=method
        self.maxiter=maxiter
        self.full_output=full_output
        self.disp=disp
        self.callback=callback
        self.return_params=return_params
        self.optim_score=optim_score
        self.optim_complex_step=optim_complex_step
        self.optim_hessian=optim_hessian
        self.flags=flags
        self.low_memory=low_memory
        self.dynamic=dynamic
        self.information_set=information_set
        self.signal_only=signal_only

        super(VARMAX, self).__init__()

    def _fit_forecaster(self, y, X=None):
        """Fit forecaster to training data.

        private method containing core logic for wrappers for statsmodel

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """

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

    def _predict(self, fh, X=None):
        """
        Wrap Statmodel's VARMAX forecast method.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasters horizon with the steps ahead to to predict.
            Default is one-step ahead forecast,
            i.e. np.array([1]) 
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        y_pred : np.ndarray
            Returns series of predicted values.
        """
        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]

        return self._fitted_forecaster.predict(
            start=start,
            end=end,
            dynamic=self.dynamic,
            information_set=self.information_set,
            signal_only=self.signal_only,
            exog=X,
        )


    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        params = [{"order": (0,0)},
                  {"order": (1,0)},
                  {"order": (0,1)},
                  {"order": (1,1)}]

        return params



