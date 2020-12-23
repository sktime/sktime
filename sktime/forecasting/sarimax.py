#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Hongyi Yang"]
__all__ = ["SARIMAX"]

import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("pmdarima")


class SARIMAX(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    """An ARIMA estimator.

    An ARIMA, or autoregressive integrated moving average, is a
    generalization of an autoregressive moving average (ARMA) and is fitted to
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
    ``ARIMA(0,1,0)`` is ``I(1)``, and ``ARIMA(0,0,1)`` is ``MA(1)``. [1]
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

            * If a string, must be a valid metric name importable from
              ``sklearn.metrics``.
            * If a callable, must adhere to the function signature::

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
    **sarimax_kwargs : keyword args, optional
        Optional arguments to pass to the SARIMAX constructor.
        Examples of potentially valuable kwargs:
          - time_varying_regression : boolean
            Whether or not coefficients on the exogenous regressors are allowed
            to vary over time.
          - enforce_stationarity : boolean
            Whether or not to transform the AR parameters to enforce
            stationarity in the auto-regressive component of the model.
          - enforce_invertibility : boolean
            Whether or not to transform the MA parameters to enforce
            invertibility in the moving average component of the model.
          - simple_differencing : boolean
            Whether or not to use partially conditional maximum likelihood
            estimation for seasonal ARIMA models. If True, differencing is
            performed prior to estimation, which discards the first
            :math:`s D + d` initial rows but results in a smaller
            state-space formulation. If False, the full SARIMAX model is
            put in state-space form so that all datapoints can be used in
            estimation. Default is False.
          - measurement_error: boolean
            Whether or not to assume the endogenous observations endog were
            measured with error. Default is False.
          - mle_regression : boolean
            Whether or not to use estimate the regression coefficients for the
            exogenous variables as part of maximum likelihood estimation or
            through the Kalman filter (i.e. recursive least squares). If
            time_varying_regression is True, this must be set to False.
            Default is True.
          - hamilton_representation : boolean
            Whether or not to use the Hamilton representation of an ARMA
            process (if True) or the Harvey representation (if False).
            Default is False.
          - concentrate_scale : boolean
            Whether or not to concentrate the scale (variance of the error
            term) out of the likelihood. This reduces the number of parameters
            estimated by maximum likelihood by one, but standard errors will
            then not be available for the scale parameter.

    References
    ----------
    https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ARIMA.html
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
    """

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
        **sarimax_kwargs
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
        self.sarimax_kwargs = sarimax_kwargs

        super(SARIMAX, self).__init__()

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
        # import inside method to avoid hard dependency
        from pmdarima.arima.arima import ARIMA as _ARIMA

        self._forecaster = _ARIMA(
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
            sarimax_kwargs=self.sarimax_kwargs,
        )

        self._set_y_X(y, X)
        self._set_fh(fh)
        self._forecaster.fit(y, X=X, **fit_params)
        self._is_fitted = True

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
            X=X,
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
            X=X,
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
        params = self._forecaster.arima_res_._results.params
        return {name: param for name, param in zip(names, params)}

    def _get_fitted_param_names(self):
        return self._forecaster.arima_res_._results.param_names
