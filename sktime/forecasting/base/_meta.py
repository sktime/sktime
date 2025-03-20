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

    # Attributes for handling heterogeneous sets of estimators
    _steps_attr = "forecasters"
    _steps_fitted_attr = "forecasters_"

    def __init__(self, forecasters, backend="loky", backend_params=None, n_jobs=None):
        if not forecasters or not isinstance(forecasters, list):
            raise ValueError(
                "'forecasters' must be a non-empty list of (name, estimator) tuples."
            )

        self.forecasters = forecasters
        self.forecasters_ = []  # Ensures it is always iterable
        self.n_jobs = n_jobs
        self.backend = backend
        self.backend_params = backend_params if backend_params else {}

        super().__init__()

    def _check_forecasters(self):
        """Validate forecasters attribute."""
        if not self.forecasters or not isinstance(self.forecasters, list):
            raise ValueError(
                "'forecasters' must be a non-empty list of (name, estimator) tuples."
            )

        try:
            names, forecasters = zip(*self.forecasters)
        except ValueError as e:
            raise ValueError(f"Invalid forecasters format: {e}")

        self._check_names(names)

        if all(est in (None, "drop") for est in forecasters):
            raise ValueError("At least one forecaster must be a valid estimator.")

        for forecaster in forecasters:
            if forecaster not in (None, "drop") and not isinstance(
                forecaster, BaseForecaster
            ):
                raise ValueError(
                    f"Invalid forecaster: {forecaster} is not a BaseForecaster."
                )

        return names, forecasters

    def _fit_forecasters(self, forecasters, y, X, fh):
        """Fit all forecasters using parallel processing."""
        if not forecasters:
            raise ValueError("forecasters cannot be None or empty")

        def _fit_single_forecaster(forecaster, meta):
            """Fit single forecaster with meta containing y, X, fh."""
            return forecaster.clone().fit(y, X, fh)

        print(f"Fitting forecasters: {forecasters}")  # Debugging print

        fitted_forecasters = parallelize(
            fun=_fit_single_forecaster,
            iter=forecasters,
            backend=self.backend,
            backend_params=self.backend_params,
        )

        if fitted_forecasters is None or not isinstance(fitted_forecasters, list):
            raise RuntimeError("Parallelized fitting returned None or an invalid type.")

        self.forecasters_ = fitted_forecasters

    def _predict_forecasters(self, fh=None, X=None):
        """Collect results from forecaster.predict() calls."""
        if not self.forecasters_:
            raise RuntimeError("forecasters_ is empty; cannot predict.")

        def _predict_single_forecaster(forecaster, meta):
            """Predict with single forecaster."""
            return forecaster.predict(fh=fh, X=X)

        return parallelize(
            fun=_predict_single_forecaster,
            iter=self.forecasters_,
            backend=self.backend,
            backend_params=self.backend_params,
        )

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
        if not self.forecasters_:
            raise RuntimeError("forecasters_ is empty; cannot update.")

        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)

        return self
