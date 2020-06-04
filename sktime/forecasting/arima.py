#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "AutoARIMA"
]

import numpy as np
import pandas as pd
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import BaseSktimeForecaster
from sktime.forecasting.base._sktime import OptionalForecastingHorizonMixin


class AutoARIMA(OptionalForecastingHorizonMixin, BaseSktimeForecaster):

    def __init__(self, start_p=2, d=None, start_q=2, max_p=5,
                 max_d=2, max_q=5, start_P=1, D=None, start_Q=1, max_P=2,
                 max_D=1, max_Q=2, max_order=5, sp=1, seasonal=True,
                 stationary=False, information_criterion='aic', alpha=0.05,
                 test='kpss', seasonal_test='ocsb', stepwise=True, n_jobs=1,
                 start_params=None, trend=None, method='lbfgs', maxiter=50,
                 offset_test_args=None, seasonal_test_args=None,
                 suppress_warnings=False, error_action='warn', trace=False,
                 random=False, random_state=None, n_fits=10,
                 out_of_sample_size=0, scoring='mse',
                 scoring_args=None, with_intercept=True,
                 **kwargs):

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
            start_p=start_p, d=d, start_q=start_q, max_p=max_p,
            max_d=max_d, max_q=max_q, start_P=start_P, D=D, start_Q=start_Q,
            max_P=max_P,
            max_D=max_D, max_Q=max_Q, max_order=max_order, m=sp,
            seasonal=seasonal,
            stationary=stationary, information_criterion=information_criterion,
            alpha=alpha,
            test=test, seasonal_test=seasonal_test, stepwise=stepwise,
            n_jobs=n_jobs,
            start_params=None, trend=trend, method=method, maxiter=maxiter,
            offset_test_args=offset_test_args,
            seasonal_test_args=seasonal_test_args,
            suppress_warnings=suppress_warnings, error_action=error_action,
            trace=trace,
            random=random, random_state=random_state, n_fits=n_fits,
            out_of_sample_size=out_of_sample_size, scoring=scoring,
            scoring_args=scoring_args, with_intercept=with_intercept,
            **kwargs
        )

    def fit(self, y_train, fh=None, X_train=None, **fit_args):
        """Fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X_train : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self._set_oh(y_train)
        self._set_fh(fh)
        self._forecaster.fit(y_train, exogenous=X_train, **fit_args)
        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        # distinguish between in-sample and out-of-sample prediction
        is_in_sample = fh <= 0
        is_out_of_sample = np.logical_not(is_in_sample)

        # pure out-of-sample prediction
        if np.all(is_out_of_sample):
            return self._predict_out_of_sample(fh, X=X,
                                               return_pred_int=return_pred_int,
                                               alpha=DEFAULT_ALPHA)

        # pure in-sample prediction
        elif np.all(is_in_sample):
            return self._predict_in_sample(fh, X=X,
                                           return_pred_int=return_pred_int,
                                           alpha=DEFAULT_ALPHA)

        # mixed in-sample and out-of-sample prediction
        else:
            fh_in_sample = fh[is_in_sample]
            fh_out_of_sample = fh[is_out_of_sample]

            y_pred_in = self._predict_in_sample(
                fh_in_sample, X=X,
                return_pred_int=return_pred_int,
                alpha=DEFAULT_ALPHA)
            y_pred_out = self._predict_out_of_sample(
                fh_out_of_sample, X=X,
                return_pred_int=return_pred_int,
                alpha=DEFAULT_ALPHA)
            return y_pred_in.append(y_pred_out)

    def _predict_in_sample(self, fh, X=None, return_pred_int=False,
                           alpha=DEFAULT_ALPHA):
        fh_abs = fh.absolute(self.cutoff)
        fh_idx = fh_abs - np.min(fh_abs)
        start = fh_abs[0]
        end = fh_abs[-1]

        if return_pred_int:

            if isinstance(alpha, (list, tuple)):
                raise NotImplementedError()
            y_pred, pred_int = self._forecaster.predict_in_sample(
                start=start,
                end=end,
                exogenous=X,
                return_conf_int=return_pred_int,
                alpha=alpha)
            y_pred = pd.Series(y_pred[fh_idx], index=fh_abs)
            pred_int = pd.DataFrame(pred_int[fh_idx, :], index=fh_abs,
                                    columns=["lower", "upper"])
            return y_pred, pred_int

        else:
            y_pred = self._forecaster.predict_in_sample(
                start=start, end=end,
                exogenous=X,
                return_conf_int=return_pred_int,
                alpha=alpha)
            return pd.Series(y_pred[fh_idx], index=fh_abs)

    def _predict_out_of_sample(self, fh, X=None, return_pred_int=False,
                               alpha=DEFAULT_ALPHA):
        # make prediction
        n_periods = int(fh[-1])
        index = fh.absolute(self.cutoff)
        fh_idx = fh.index_like(self.cutoff)

        if return_pred_int:
            y_pred, pred_int = self._forecaster.model_.predict(
                n_periods=n_periods, exogenous=X,
                return_conf_int=return_pred_int, alpha=alpha)
            y_pred = pd.Series(y_pred[fh_idx], index=index)
            pred_int = pd.DataFrame(pred_int[fh_idx, :], index=index,
                                    columns=["lower", "upper"])
            return y_pred, pred_int
        else:
            y_pred = self._forecaster.model_.predict(
                n_periods=n_periods,
                exogenous=X,
                return_conf_int=return_pred_int,
                alpha=alpha)
            return pd.Series(y_pred[fh_idx], index=index)

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
