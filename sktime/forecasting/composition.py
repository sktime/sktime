#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["TransformedTargetForecaster"]

from sklearn.base import clone
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base import DEFAULT_ALPHA
from sktime.forecasting.base import MetaForecasterMixin


class TransformedTargetForecaster(BaseForecaster, MetaForecasterMixin):
    """Meta-estimator for forecasting transformed time series."""

    _required_parameters = ["forecaster", "transformer"]

    def __init__(self, forecaster, transformer):
        self.forecaster = forecaster
        self.transformer = transformer
        self.transformer_ = clone(self.transformer)
        self.forecaster_ = clone(self.forecaster)
        super(TransformedTargetForecaster, self).__init__()

    def fit(self, y_train, fh=None, X_train=None):
        yt = self.transform(y_train)
        self.forecaster_.fit(yt, fh=fh, X_train=X_train)
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()
        self._check_is_fitted()
        y_pred = self.forecaster_.predict(fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha)
        yit = self.inverse_transform(y_pred)
        return yit

    def update(self, y_new, X_new=None, update_params=False):
        self._check_is_fitted()
        if hasattr(self.transformer_, "update"):
            self.transformer_.update(y_new, X_new=X_new, update_params=update_params)
        self.forecaster_.update(y_new, X_new=X_new, update_params=update_params)
        return self

    def update_predict(self, y_test, cv=None, X_test=None, update_params=False, return_pred_int=False,
                       alpha=DEFAULT_ALPHA):
        return self.forecaster_.update_predict(y_test, cv=cv, X_test=X_test, update_params=update_params,
                                               return_pred_int=return_pred_int, alpha=alpha)

    def transform(self, y):
        return self.transformer_.fit_transform(y)

    def inverse_transform(self, y):
        return self.transformer_.inverse_transform(y)

    def _set_fh(self, fh):
        self.forecaster_._set_fh(fh)

    def _set_oh(self, oh):
        self.forecaster_._set_oh(oh)

    @property
    def now(self):
        return self.forecaster_.now

    @property
    def oh(self):
        return self.forecaster_.oh

    @property
    def fh(self):
        return self.forecaster_.fh
