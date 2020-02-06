#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = [
    "Detrender"
]
__author__ = ["Markus Löning"]


from sklearn.base import clone
from sktime.forecasting.base import _BaseTemporalEstimator
from sktime.utils.validation.forecasting import validate_y


class _BaseDetrender(_BaseTemporalEstimator):

    def fit(self, y_train, fh=None):
        raise NotImplementedError()

    def transform(self, y):
        raise NotImplementedError()

    def inverse_transform(self, y):
        raise NotImplementedError()

    def fit_transform(self, y_train):
        return self.fit(y_train).transform(y_train)

    def update(self, y_new):
        raise NotImplementedError()


class Detrender(_BaseDetrender):

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.forecaster_ = None
        super(Detrender, self).__init__()

    def fit(self, y_train, fh=None):
        y_train = validate_y(y_train)
        self._set_obs_horizon(y_train.index)

        forecaster = clone(self.forecaster)

        self.forecaster_ = forecaster.fit(y_train, fh=fh)

        self._is_fitted = True
        return self

    def transform(self, y):
        y = validate_y(y)
        self._check_is_fitted()

        fh = self._get_fh(y)
        y_pred = self.forecaster_.predict(fh)
        residuals = y - y_pred

        return residuals

    def _get_fh(self, y):
        # infer fh from obs_horizon and index of passed `y`
        fh = y.index - self._now
        return fh.values

    def inverse_transform(self, y):
        y = validate_y(y)
        self._check_is_fitted()

        fh = self._get_fh(y)
        y_pred = self.forecaster_.predict(fh)
        residuals = y + y_pred

        return residuals
