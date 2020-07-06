#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "MetaForecasterMixin",
    "BaseHeterogenousEnsembleForecaster"
]

from joblib import Parallel
from joblib import delayed
from sklearn.base import clone
from sktime.base import BaseHeterogenousMetaEstimator
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._base import is_forecaster
from sktime.forecasting.base._sktime import BaseSktimeForecaster


class MetaForecasterMixin:
    """Mixin class for all meta forecasters in sktime."""
    _required_parameters = ["forecaster"]


class BaseHeterogenousEnsembleForecaster(BaseSktimeForecaster,
                                         BaseHeterogenousMetaEstimator):
    """Base class for heterogenous ensemble forecasters"""
    _required_parameters = ["forecasters"]

    def __init__(self, forecasters, n_jobs=None):
        self.forecasters = forecasters
        self.forecasters_ = None
        self.n_jobs = n_jobs
        super(BaseHeterogenousEnsembleForecaster, self).__init__()

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
            if forecaster not in (None, 'drop') and not is_forecaster(
                    forecaster):
                raise ValueError(
                    "The estimator {} should be a {}.".format(
                        forecaster.__class__.__name__,
                        is_forecaster.__name__[3:]
                    )
                )
        return names, forecasters

    def _fit_forecasters(self, forecasters, y_train, fh=None, X_train=None):
        """Fit all forecasters in parallel"""

        def _fit_forecaster(forecaster, y_train, fh, X_train):
            """Fit single forecaster"""
            return forecaster.fit(y_train, fh=fh, X_train=X_train)

        self.forecasters_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_forecaster)(clone(forecaster), y_train, fh, X_train)
            for forecaster in forecasters)

    def _predict_forecasters(self, fh=None, X=None, return_pred_int=False,
                             alpha=DEFAULT_ALPHA):
        """Collect results from forecaster.predict() calls."""
        if return_pred_int:
            raise NotImplementedError()
        # return Parallel(n_jobs=self.n_jobs)(delayed(forecaster.predict)(
        # fh, X=X)
        #                                     for forecaster in
        #                                     self.forecasters_)
        return [forecaster.predict(fh=fh, X=X, return_pred_int=return_pred_int,
                                   alpha=alpha)
                for forecaster in self.forecasters_]

    def get_params(self, deep=True):
        return self._get_params("forecasters", deep=deep)

    def set_params(self, **params):
        self._set_params("forecasters", **params)
        return self
