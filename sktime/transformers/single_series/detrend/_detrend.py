#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = [
    "Detrender"
]
__author__ = ["Markus Löning"]

from sklearn.base import clone
from sktime.forecasting.base._meta import MetaForecasterMixin
from sktime.transformers.single_series.base import \
    BaseSingleSeriesTransformer
from sktime.utils.validation.forecasting import check_y


class Detrender(MetaForecasterMixin, BaseSingleSeriesTransformer):

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.forecaster_ = None
        super(Detrender, self).__init__()

    def fit(self, y_train, X_train=None):
        forecaster = clone(self.forecaster)
        self.forecaster_ = forecaster.fit(y_train, X_train=X_train)
        self._is_fitted = True
        return self

    def transform(self, y, X=None):
        self.check_is_fitted()
        y = check_y(y)
        fh = self._get_relative_fh(y)
        y_pred = self.forecaster_.predict(fh=fh, X=X)
        return y - y_pred

    def inverse_transform(self, y, X=None):
        self.check_is_fitted()
        y = check_y(y)
        fh = self._get_relative_fh(y)
        y_pred = self.forecaster_.predict(fh=fh, X=X)
        return y + y_pred

    def _get_relative_fh(self, y):
        return y.index.values - self.forecaster_.cutoff

    def update(self, y_new, update_params=False):
        """Update fitted parameters

         Parameters
         ----------
         y_new : pd.Series
         update_params : bool, optional (default=False)

         Returns
         -------
         self : an instance of self
         """
        self.forecaster_.update(y_new, update_params=update_params)
        return self
