#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements ensemble forecasters.

Creates univariate (optionally weighted) combination of the predictions from underlying
forecasts.
"""

__author__ = ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"]
__all__ = ["EnsembleForecaster", "AutoEnsembleForecaster"]

import warnings

import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.pipeline import Pipeline

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.stats import (
    _weighted_geometric_mean,
    _weighted_max,
    _weighted_median,
    _weighted_min,
)
from sktime.utils.validation.forecasting import check_regressor

VALID_AGG_FUNCS = {
    "mean": {"unweighted": np.mean, "weighted": np.average},
    "median": {"unweighted": np.median, "weighted": _weighted_median},
    "min": {"unweighted": np.min, "weighted": _weighted_min},
    "max": {"unweighted": np.max, "weighted": _weighted_max},
    "gmean": {"unweighted": gmean, "weighted": _weighted_geometric_mean},
}


class AutoEnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Automatically find best weights for the ensembled forecasters."""

    _tags = {
        "authors": ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"],
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "scitype:y": "univariate",
    }

    def __init__(
        self,
        forecasters,
        method="feature-importance",
        regressor=None,
        test_size=None,
        random_state=None,
        n_jobs=None,
        backend="loky",
        backend_params=None,
    ):
        if n_jobs is not None:
            warnings.warn(
                "`n_jobs` is deprecated and will be removed in a future version. "
                "Use `backend` instead.",
                FutureWarning,
            )
            backend = backend or "locky"  # use default backend if not set
            backend_params = backend_params or {"n_jobs": n_jobs}

        super().__init__(
            forecasters=forecasters,
            backend=backend,
            backend_params=backend_params,
        )
        self.method = method
        self.regressor = regressor
        self.test_size = test_size
        self.random_state = random_state

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional, default=None
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        _, forecasters = self._check_forecasters()

        # get training data for meta-model
        if X is not None:
            y_train, y_test, X_train, X_test = temporal_train_test_split(
                y, X, test_size=self.test_size
            )
        else:
            y_train, y_test = temporal_train_test_split(y, test_size=self.test_size)
            X_train, X_test = None, None

        # fit ensemble models
        fh_test = ForecastingHorizon(y_test.index, is_relative=False)
        self._fit_forecasters(forecasters, y_train, X_train, fh_test)

        if self.method == "feature-importance":
            self.regressor_ = check_regressor(
                regressor=self.regressor, random_state=self.random_state
            )
            X_meta = pd.concat(self._predict_forecasters(fh_test, X_test), axis=1)
            X_meta.columns = pd.RangeIndex(len(X_meta.columns))

            # fit meta-model (regressor) on predictions of ensemble models
            self.regressor_.fit(X=X_meta, y=y_test)

            # check if regressor is a sklearn.Pipeline
            if isinstance(self.regressor_, Pipeline):
                # extract regressor from pipeline to access its attributes
                self.weights_ = _get_weights(self.regressor_.steps[-1][1])
            else:
                self.weights_ = _get_weights(self.regressor_)

        elif self.method == "inverse-variance":
            # get in-sample forecasts
            if self.regressor is not None:
                Warning(f"regressor will not be used because ${self.method} is set.")
            inv_var = np.array(
                [
                    1 / np.var(y_test - y_pred_test)
                    for y_pred_test in self._predict_forecasters(fh_test, X_test)
                ]
            )
            # standardize the inverse variance
            self.weights_ = list(inv_var / np.sum(inv_var))
        else:
            raise NotImplementedError(
                f"Given method {self.method} does not exist, "
                f"please provide valid method parameter."
            )

        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _predict(self, fh, X):
        """Return the predicted reduction.

        Parameters
        ----------
        fh : int, list or np.array, optional, default=None
        X : pd.DataFrame

        Returns
        -------
        y_pred : pd.Series
            Aggregated predictions.
        """
        y_pred_df = pd.concat(self._predict_forecasters(fh, X), axis=1)
        # apply weights
        y_pred = y_pred_df.apply(lambda x: np.average(x, weights=self.weights_), axis=1)
        y_pred.name = self._y.name
        return y_pred


class EnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Ensemble of forecasters."""

    _tags = {
        "authors": ["mloning", "GuzalBulatova", "aiwalter", "RNKuhns", "AnH0ang"],
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "both",
    }

    _steps_attr = "_forecasters"
    _steps_fitted_attr = "forecasters_"

    def __init__(
        self,
        forecasters,
        n_jobs=None,
        backend="loky",
        backend_params=None,
        aggfunc="mean",
        weights=None,
    ):
        if n_jobs is not None:
            warnings.warn(
                "`n_jobs` is deprecated and will be removed in a future version. "
                "Use `backend` instead.",
                FutureWarning,
            )
            backend = backend or "loky"
            backend_params = backend_params or {"n_jobs": n_jobs}

        self.aggfunc = aggfunc
        self.weights = weights
        super().__init__(
            forecasters=forecasters, backend=backend, backend_params=backend_params
        )

        fc = []
        for forecaster in forecasters:
            if len(forecaster) <= 2:
                # Handle the (str, est) tuple
                fc.append(forecaster)
            elif len(forecaster) == 3:
                # Handle the (str, est, num_replicates) tuple
                name, estimator, num_replicates = forecaster
                fc.extend([(name, estimator)] * num_replicates)
            else:
                msg = (
                    "Error in EnsembleForecaster construction: "
                    "forecasters argument must be as list of "
                    "estimator, (str, estimator) or (str, estimator, count) tuples."
                )
                raise ValueError(msg)

        self._forecasters = self._check_estimators(
            fc, clone_ests=False, allow_empty=True
        )
        self.forecasters_ = self._check_estimators(
            fc, clone_ests=True, allow_empty=True
        )

        # the ensemble requires fh in fit
        # iff any of the component forecasters require fh in fit
        self._anytagis_then_set("requires-fh-in-fit", True, False, self._forecasters)

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.DataFrame - Series, Panel, or Hierarchical mtype format.
            Target time series to which to fit the forecaster.
        fh : ForecastingHorizon, optional, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional, default=None, must be of same mtype as y
            Exogenous data to which to fit the forecaster.

        Returns
        -------
        self : returns an instance of self.
        """
        forecasters = [f[1] for f in self._forecasters]
        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _predict(self, fh, X):
        """Return the predicted reduction.

        Parameters
        ----------
        fh : ForecastingHorizon, optional, default=None
        X : pd.DataFrame, optional, default=None, must be of same mtype as y
            Exogenous data to which to fit the forecaster.

        Returns
        -------
        y_pred : pd.DataFrame - Series, Panel, or Hierarchical mtype format,
            will be of same mtype as y in _fit
            Ensembled predictions
        """
        names = [f[0] for f in self._forecasters]
        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1, keys=names)
        y_pred = (
            y_pred.T.groupby(level=1)
            .agg(
                lambda y, aggfunc, weights: _aggregate(y.T, aggfunc, weights),
                self.aggfunc,
                self.weights,
            )
            .T
        )
        return y_pred


# Helper functions remain unchanged
def _get_weights(regressor):
    if hasattr(regressor, "feature_importances_"):
        weights = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        weights = regressor.coef_
    else:
        raise NotImplementedError(
            """The given regressor is not supported. It must have
            either an attribute feature_importances_ or coef_ after fitting."""
        )
    # avoid ZeroDivisionError if all weights are 0
    if weights.sum() == 0:
        weights += 1
    return list(weights)


def _aggregate(y, aggfunc, weights):
    """Apply aggregation function by row."""
    if weights is None:
        aggfunc = _check_aggfunc(aggfunc, weighted=False)
        y_agg = aggfunc(y, axis=1)
    else:
        aggfunc = _check_aggfunc(aggfunc, weighted=True)
        y_agg = aggfunc(y, axis=1, weights=np.array(weights))

    return pd.Series(y_agg, index=y.index)


def _check_aggfunc(aggfunc, weighted=False):
    _weighted = "weighted" if weighted else "unweighted"
    if aggfunc not in VALID_AGG_FUNCS.keys():
        raise ValueError("Aggregation function %s not recognized." % aggfunc)
    return VALID_AGG_FUNCS[aggfunc][_weighted]
