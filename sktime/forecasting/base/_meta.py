#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["_HeterogenousEnsembleForecaster"]

from joblib import Parallel
from joblib import delayed
from sklearn.base import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._base import BaseForecaster


class _HeterogenousEnsembleForecaster(BaseForecaster, _HeterogenousMetaEstimator):
    """Base class for heterogenous ensemble forecasters"""

    _required_parameters = ["forecasters"]

    def __init__(self, forecasters, n_jobs=None):
        self.forecasters = forecasters
        self.forecasters_ = None
        self.n_jobs = n_jobs
        super(_HeterogenousEnsembleForecaster, self).__init__()

    def _check_forecasters(self):
        if (
            self.forecasters is None
            or len(self.forecasters) == 0
            or not isinstance(self.forecasters, list)
        ):
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator) tuples."
            )
        names, forecasters = zip(*self.forecasters)
        # defined by MetaEstimatorMixin
        self._check_names(names)

        has_estimator = any(est not in (None, "drop") for est in forecasters)
        if not has_estimator:
            raise ValueError(
                "All estimators are dropped. At least one is required "
                "to be an estimator."
            )

        for forecaster in forecasters:
            if forecaster not in (None, "drop") and not isinstance(
                forecaster, BaseForecaster
            ):
                raise ValueError(
                    f"The estimator {forecaster.__class__.__name__} should be a "
                    f"Forecaster."
                )
        return names, forecasters

    def _fit_forecasters(self, forecasters, y, X, fh):
        """Fit all forecasters in parallel"""

        def _fit_forecaster(forecaster, y, X, fh):
            """Fit single forecaster"""
            return forecaster.fit(y, X, fh)

        self.forecasters_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_forecaster)(clone(forecaster), y, X, fh)
            for forecaster in forecasters
        )

    def _predict_forecasters(
        self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Collect results from forecaster.predict() calls."""
        if return_pred_int:
            raise NotImplementedError()

        return [
            forecaster.predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)
            for forecaster in self.forecasters_
        ]

    def get_params(self, deep=True):
        return self._get_params("forecasters", deep=deep)

    def set_params(self, **params):
        self._set_params("forecasters", **params)
        return self
