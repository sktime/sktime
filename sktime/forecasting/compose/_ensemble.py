#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["EnsembleForecaster"]

from joblib import Parallel, delayed
from sklearn.base import clone
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base import DEFAULT_ALPHA
from sktime.forecasting.base import MetaForecasterMixin
from sktime.forecasting.base import is_forecaster
from sktime.forecasting.base import OptionalForecastingHorizonMixin
import pandas as pd
import numpy as np


class EnsembleForecaster(OptionalForecastingHorizonMixin, MetaForecasterMixin, BaseForecaster):

    _required_parameters = ["forecasters"]

    def __init__(self, forecasters, n_jobs=None):
        self.forecasters = forecasters
        self.forecasters_ = None
        self.n_jobs = n_jobs
        super(EnsembleForecaster, self).__init__()

    def fit(self, y_train, fh=None, X_train=None):
        self._set_oh(y_train)
        self._set_fh(fh)
        names, forecasters = self._check_forecasters()

        self.forecasters_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_forecaster)(clone(forecaster), y_train, X_train=X_train)
            for forecaster in forecasters)
        self._is_fitted = True
        return self

    def update(self, y_new, X_new=None, update_params=False):
        self._check_is_fitted()
        self._set_oh(y_new)

        for forecaster in self.forecasters_:
            forecaster.update(y_new, X_new=X_new, update_params=update_params)

        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        self._check_is_fitted()
        self._set_fh(fh)
        if return_pred_int:
            raise NotImplementedError()
        return self._predict(self.fh, X=X).mean(axis=1)

    def _predict(self, fh=None, X=None):
        """Collect results from forecaster.predict() calls."""
        return pd.concat([forecaster.predict(fh, X=X) for forecaster in self.forecasters_], axis=1)

    def transform(self, fh=None, X=None):
        self._check_is_fitted()
        self._set_fh(fh)
        return self._predict(fh=self.fh, X=X)

    def _check_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))

    def _check_forecasters(self):
        if self.forecasters is None or len(self.forecasters) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator) tuples."
            )
        names, forecasters = zip(*self.forecasters)
        # defined by MetaEstimatorMixin
        self._check_names(names)

        has_estimator = any(est not in (None, 'drop') for est in forecasters)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )

        for forecaster in forecasters:
            if forecaster not in (None, 'drop') and not is_forecaster(forecaster):
                raise ValueError(
                    "The estimator {} should be a {}.".format(
                        forecaster.__class__.__name__, is_forecaster.__name__[3:]
                    )
                )
        return names, forecasters


def _fit_forecaster(forecaster, y_train, X_train=None):
    return forecaster.fit(y_train, X_train=X_train)
