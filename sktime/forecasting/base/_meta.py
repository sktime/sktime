#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements meta forecaster for forecasters composed of other estimators."""

__author__ = ["mloning"]
__all__ = ["_HeterogenousEnsembleForecaster"]

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import BaseForecaster
from sktime.registry import is_scitype


class _HeterogenousEnsembleForecaster(_HeterogenousMetaEstimator, BaseForecaster):
    """Base class for heterogeneous ensemble forecasters."""

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_forecasters"

    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "forecasters_"

    def __init__(self, forecasters, n_jobs=None, fc_alt=None):
        if forecasters is not None:
            self.forecasters = forecasters
        self.n_jobs = n_jobs
        super().__init__()

        if fc_alt is not None:
            fc = fc_alt
        else:
            fc = forecasters

        if fc is not None:
            self._initialize_forecaster_tuples(fc)

    def _initialize_forecaster_tuples(self, forecasters):
        """Initialize estimator tuple attributes, default.

        Initializes:

        - self.forecasters_ from self.forecasters, this is coerced to a list of tuples
        - self._forecasters from self.forecasters, same as above but also cloned

        Parameters
        ----------
        forecasters : list of estimators, or list of (name, estimator) pairs
        """
        self._forecasters = self._check_forecasters_init(forecasters)
        self.forecasters_ = self._check_estimators(
            forecasters, clone_ests=True, allow_empty=True
        )

    def _check_forecasters_init(self, estimators):
        """Check Steps.

        Parameters
        ----------
        estimators : list of estimators, or list of (name, estimator) pairs
        allow_postproc : bool, optional, default=False
            whether transformers after the forecaster are allowed

        Returns
        -------
        step : list of (name, estimator) pairs, estimators are cloned (not references)
            if estimators was a list of (str, estimator) tuples, then just cloned
            if was a list of estimators, then str are generated via _get_estimator_names

        Raises
        ------
        TypeError if names in ``estimators`` are not unique
        TypeError if estimators in ``estimators`` are not all forecaster or transformer
        TypeError if there is not exactly one forecaster in ``estimators``
        TypeError if not allow_postproc and forecaster is not last estimator
        """
        self_name = type(self).__name__
        if not isinstance(estimators, (list, tuple)):
            estimators = [estimators]

        estimator_tuples = self._get_estimator_tuples(estimators, clone_ests=False)

        estimators = [x[1] for x in estimator_tuples]

        # validate names
        if not all([is_scitype(x, "forecaster") for x in estimators]):
            raise TypeError(f"estimators passed to {self_name} must be forecasters")

        return estimator_tuples

    def _get_forecaster_list(self):
        """Return list of forecasters."""
        return [x[1] for x in self.forecasters_]

    def _get_forecaster_names(self):
        """Return list of forecaster names."""
        return [x[0] for x in self.forecasters_]

    def _fit_forecasters(self, forecasters, y, X, fh):
        """Fit all forecasters in parallel.

        Returns
        -------
        list of references to fitted forecasters
            in same order as forecasters
        """
        from joblib import Parallel, delayed

        def _fit_forecaster(forecaster, y, X, fh):
            """Fit single forecaster."""
            return forecaster.fit(y, X, fh)

        if forecasters is None:
            forecasters = self._get_forecaster_list()

        fitted_fcst = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_forecaster)(forecaster.clone(), y, X, fh)
            for forecaster in forecasters
        )
        fcst_names = self._get_forecaster_names()
        self.forecasters_ = list(zip(fcst_names, fitted_fcst))

    def _predict_forecasters(self, fh=None, X=None, forecasters=None):
        """Collect results from forecaster.predict() calls."""
        if forecasters is None:
            forecasters = self._get_forecaster_list()
        return [forecaster.predict(fh=fh, X=X) for forecaster in forecasters]

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
        for forecaster in self._get_forecaster_list():
            forecaster.update(y, X, update_params=update_params)
        return self
