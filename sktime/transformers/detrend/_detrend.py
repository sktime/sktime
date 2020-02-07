#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = [
    "Detrender"
]
__author__ = ["Markus Löning"]


from sklearn.base import clone
from sktime.forecasting.base import _BaseTemporalEstimator
from sktime.utils.validation.forecasting import check_y
import pandas as pd
import numpy as np


class Detrender(_BaseTemporalEstimator):

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.forecaster_ = None
        super(Detrender, self).__init__()

    def fit(self, y_train, X_train=None):
        # input checks
        y_train = check_y(y_train)
        self._set_obs_horizon(y_train.index)

        # fit forecaster
        # forecasters which require fh in fit are not supported
        forecaster = clone(self.forecaster)
        self.forecaster_ = forecaster.fit(y_train, X_train=X_train)

        self._is_fitted = True
        return self

    def fit_transform(self, y_train, X_train=None):
        # first fit
        self.fit(y_train, X_train=X_train)

        # then directly transform via in-sample prediction
        y_pred = self.forecaster_.predict_in_sample(y_train, X_train=X_train)
        residuals = y_train - y_pred
        return residuals

    def transform(self, y, X=None):
        y = check_y(y)
        self._check_is_fitted()

        # get predictions from forecaster
        y_pred = self._predict(y, X=X)

        # remove trend from series
        residuals = y - y_pred
        return residuals

    def inverse_transform(self, y, X=None):
        y = check_y(y)
        self._check_is_fitted()

        # get predictions from forecaster
        y_pred = self._predict(y, X=X)

        # add trend back to series
        yit = y + y_pred
        return yit

    def _predict(self, y, X=None):
        """Make in-sample or out-of-sample predictions depending on
        passed time series and known observation horizon"""

        # check if time index is in observation horizon seen in training
        isin_obs_horizon = y.index.isin(self._obs_horizon)

        # raise error if y contains values before observation horizon
        # seen in training
        if np.any(y.index.values < self._obs_horizon.min()):
            raise ValueError("Passed `y` contains values from before the "
                             "observation horizon seen in `fit`")

        # if all values are in observation horizon,
        # make in-sample forecasts
        if isin_obs_horizon.all():
            fh = y.index.values - self._obs_horizon.min() + 1
            y_pred = self.forecaster_.predict_in_sample(y, fh=fh, X_train=X)

        # if only some of the values are in the observation horizon,
        # make both in-sample and out-of-sample predictions
        elif isin_obs_horizon.any():

            if X is not None:
                # split x according to in-sample or out-of-sample predictions
                raise NotImplementedError()

            # in-sample predictions
            in_sample = y.index[isin_obs_horizon]
            fh = in_sample.values
            y_train = y.iloc[in_sample]
            in_pred = self.forecaster_.predict_in_sample(y_train, fh=fh, X_train=None)

            # out-of-sample predictions
            out_of_sample = y.index[~isin_obs_horizon]
            fh = out_of_sample.values - self._obs_horizon.max()
            out_pred = self.forecaster_.predict(fh=fh, X=None)

            # combine predictions
            y_pred = in_pred.append(out_pred)

        # if all values are out-of-sample, call predict
        else:
            # get relative forecasting horizon
            fh = y.index.values - self.now
            y_pred = self.forecaster_.predict(fh=fh, X=X)

        return pd.Series(y_pred.values, index=y.index)

