#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Implements meta forecaster for forecasters composed of other estimators."""

__author__ = ["mloning"]
__all__ = ["_HeterogenousEnsembleForecaster"]

from joblib import Parallel, delayed

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import BaseForecaster


class _HeterogenousEnsembleForecaster(BaseForecaster, _HeterogenousMetaEstimator):
    """Base class for heterogeneous ensemble forecasters."""

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
        """Fit all forecasters in parallel."""

        def _fit_forecaster(forecaster, y, X, fh):
            """Fit single forecaster."""
            return forecaster.fit(y, X, fh)

        self.forecasters_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_forecaster)(forecaster.clone(), y, X, fh)
            for forecaster in forecasters
        )

    def _predict_forecasters(self, fh=None, X=None):
        """Collect results from forecaster.predict() calls."""
        return [forecaster.predict(fh=fh, X=X) for forecaster in self.forecasters_]

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional, default=True

        Returns
        -------
        self : an instance of self.
        """
        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)
        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("forecasters", deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params("forecasters", **params)
        return self
