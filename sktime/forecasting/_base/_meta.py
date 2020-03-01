#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = [
    "MetaForecasterMixin",
    "BaseHeterogenousMetaForecaster"
]

from joblib import Parallel, delayed
from sklearn.base import clone
from sktime.forecasting._base import BaseForecaster
from sktime.forecasting._base import is_forecaster


class MetaForecasterMixin:
    """Mixin class for all meta forecasters in sktime."""
    _required_parameters = ["forecaster"]


class BaseHeterogenousMetaForecaster(MetaForecasterMixin, BaseForecaster):
    """Base class for heterogenous ensemble forecasters"""
    _required_parameters = ["forecasters"]

    def __init__(self, forecasters, n_jobs=None):
        self.forecasters = forecasters
        self.forecasters_ = None
        self.n_jobs = n_jobs
        super(BaseHeterogenousMetaForecaster, self).__init__()

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

    def _fit_forecasters(self, forecasters, y_train, X_train):
        """Helper function to fit all forecasters"""

        def _fit_forecaster(forecaster, y_train, X_train):
            """Helper function to fit single forecaster"""
            return forecaster.fit(y_train, X_train=X_train)

        self.forecasters_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_forecaster)(clone(forecaster), y_train, X_train)
            for forecaster in forecasters)

    def _predict_forecasters(self, fh=None, X=None):
        """Collect results from forecaster.predict() calls."""
        # return Parallel(n_jobs=self.n_jobs)(delayed(forecaster.predict)(fh, X=X)
        #                                     for forecaster in self.forecasters_)
        return [forecaster.predict(fh, X=X) for forecaster in self.forecasters_]
