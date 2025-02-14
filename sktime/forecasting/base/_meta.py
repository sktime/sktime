#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements meta forecaster for forecasters composed of other estimators."""

__author__ = ["mloning"]
__all__ = ["_HeterogenousEnsembleForecaster"]

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import BaseForecaster
from sktime.utils.parallel import parallelize


class _HeterogenousEnsembleForecaster(_HeterogenousMetaEstimator, BaseForecaster):
    """Base class for heterogeneous ensemble forecasters."""

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # *steps*attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "forecasters"

    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, *steps*fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "forecasters_"

    def __init__(self, forecasters, backend="loky", backend_params=None):
        self.forecasters = forecasters
        self.forecasters_ = None
        self.n_jobs = None
        self.backend = backend
        self.backend_params = backend_params
        super().__init__()

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
        """Fit all forecasters using parallel processing."""

        def _fit_single_forecaster(forecaster, meta):
            """Fit single forecaster with meta containing y, X, fh."""
            return forecaster.clone().fit(meta["y"], meta["X"], meta["fh"])

        self.forecasters_ = parallelize(
            fun=_fit_single_forecaster,
            iter=forecasters,
            meta={"y": y, "X": X, "fh": fh},
            backend=self.backend,
            backend_params=self.backend_params,
        )

    def _predict_forecasters(self, fh=None, X=None):
        """Collect results from forecaster.predict() calls."""

        def _predict_single_forecaster(forecaster, fh, X):
            """Predict with single forecaster."""
            return forecaster.predict(fh=fh, X=X)

        return parallelize(
            fun=_predict_single_forecaster,
            iter=self.forecasters_,
            backend=self.backend,
            backend_params=self.backend_params,
        )

    def update(self, y, X=None, update_params=True):
        """Update fitted parameters.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional, default=True

        Returns
        -------
        self : an instance of self
        """

        def _update_single_forecaster(forecaster, y, X, update_params):
            """Update single forecaster."""
            return forecaster.update(y, X, update_params=update_params)

        self.forecasters_ = parallelize(
            fun=_update_single_forecaster,
            iter=self.forecasters_,
            backend=self.backend,
            backend_params=self.backend_params,
            update_params=update_params,
        )
        return self
