#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = [
    "AutoARIMAForecaster"
]

from pmdarima.arima import AutoARIMA
from sktime.forecasting._base import BaseForecaster
from sktime.forecasting._base import OptionalForecastingHorizonMixin
from sktime.forecasting._base import DEFAULT_ALPHA
import pandas as pd
import numpy as np


class AutoARIMAForecaster(OptionalForecastingHorizonMixin, BaseForecaster):

    def __init__(self, start_p=2, d=None, start_q=2, max_p=5,
                 max_d=2, max_q=5, start_P=1, D=None, start_Q=1, max_P=2,
                 max_D=1, max_Q=2, max_order=5, m=1, seasonal=True,
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
        self.m = m
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

        super(AutoARIMAForecaster, self).__init__()

        self.model = AutoARIMA(
            start_p=start_p, d=d, start_q=start_q, max_p=max_p,
            max_d=max_d, max_q=max_q, start_P=start_P, D=D, start_Q=start_Q, max_P=max_P,
            max_D=max_D, max_Q=max_Q, max_order=max_order, m=m, seasonal=seasonal,
            stationary=stationary, information_criterion=information_criterion, alpha=alpha,
            test=test, seasonal_test=seasonal_test, stepwise=stepwise, n_jobs=n_jobs,
            start_params=None, trend=trend, method=method, maxiter=maxiter,
            offset_test_args=offset_test_args, seasonal_test_args=seasonal_test_args,
            suppress_warnings=suppress_warnings, error_action=error_action, trace=trace,
            random=random, random_state=random_state, n_fits=n_fits,
            out_of_sample_size=out_of_sample_size, scoring=scoring,
            scoring_args=scoring_args, with_intercept=with_intercept,
            **kwargs
        )

    def fit(self, y_train, fh=None, X_train=None, **fit_args):
        self._set_oh(y_train)
        self._set_fh(fh)
        self.model.fit(y_train, exogenous=X_train, **fit_args)
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        self._check_is_fitted()
        self._set_fh(fh)

        # distinguish between in-sample and out-of-sample prediction
        is_in_sample = self.fh <= 0
        is_out_of_sample = np.logical_not(is_in_sample)

        # pure out-of-sample prediction
        if np.all(is_out_of_sample):
            return self._predict_out_of_sample(self.fh, X=X, return_pred_int=return_pred_int, alpha=DEFAULT_ALPHA)

        # pure in-sample prediction
        elif np.all(is_in_sample):
            return self._predict_in_sample(self.fh, X=X, return_pred_int=return_pred_int, alpha=DEFAULT_ALPHA)

        # mixed in-sample and out-of-sample prediction
        else:
            fh_in_sample = self.fh[is_in_sample]
            fh_out_of_sample = self.fh[is_out_of_sample]

            y_pred_in = self._predict_in_sample(fh_in_sample, X=X, return_pred_int=return_pred_int,
                                                alpha=DEFAULT_ALPHA)
            y_pred_out = self._predict_out_of_sample(fh_out_of_sample, X=X, return_pred_int=return_pred_int,
                                                     alpha=DEFAULT_ALPHA)
            return y_pred_in.append(y_pred_out)

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        fh_abs = self._get_absolute_fh(fh)
        fh_idx = fh_abs - np.min(fh_abs)
        start = fh_abs[0]
        end = fh_abs[-1]

        if return_pred_int:
            y_pred, pred_int = self.model.model_.predict_in_sample(start=start, end=end, exogenous=X,
                                                         return_conf_int=return_pred_int, alpha=alpha)
            return pd.Series(y_pred[fh_idx], index=fh_abs), pd.DataFrame(pred_int[fh_idx, :], index=fh_abs)

        else:
            y_pred = self.model.model_.predict_in_sample(start=start, end=end, exogenous=X,
                                                         return_conf_int=return_pred_int, alpha=alpha)
            return pd.Series(y_pred[fh_idx], index=fh_abs)

    def _predict_out_of_sample(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        # make prediction
        n_periods = int(fh[-1])
        index = self._get_absolute_fh(fh)
        fh_idx = self._get_array_index_fh(fh)

        if return_pred_int:
            y_pred, pred_int = self.model.model_.predict(n_periods=n_periods, exogenous=X,
                                                         return_conf_int=return_pred_int, alpha=alpha)
            return pd.Series(y_pred[fh_idx], index=index), pd.DataFrame(pred_int[fh_idx, :], index=index)

        else:
            y_pred = self.model.model_.predict(n_periods=n_periods, exogenous=X, return_conf_int=return_pred_int,
                                               alpha=alpha)
            return pd.Series(y_pred[fh_idx], index=index)

    def update(self, y_new, X_new=None, update_params=False):
        self._check_is_fitted()
        self._set_oh(y_new)
        if update_params:
            raise NotImplementedError()
        return self

    def get_fitted_params(self):
        self._check_is_fitted()
        names = self.get_fitted_param_names()
        params = self.model.model_.arima_res_._results.params
        return {name: param for name, param in zip(names, params)}

    def get_fitted_param_names(self):
        self._check_is_fitted()
        return self.model.model_.arima_res_._results.param_names



