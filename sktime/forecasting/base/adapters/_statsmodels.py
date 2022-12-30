# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for statsmodels forecasters to be used in sktime framework."""

__author__ = ["mloning"]
__all__ = ["_StatsModelsAdapter"]

import inspect
from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class _StatsModelsAdapter(BaseForecaster):
    """Base class for interfacing statsmodels forecasting algorithms."""

    _fitted_param_names = ()
    _tags = {
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "python_dependencies": "statsmodels",
    }

    def __init__(self, random_state=None):
        self._forecaster = None
        self.random_state = random_state
        self._fitted_forecaster = None
        super(_StatsModelsAdapter, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored

        Returns
        -------
        self : returns an instance of self.
        """
        # statsmodels does not support the pd.Int64Index as required,
        # so we coerce them here to pd.RangeIndex
        if isinstance(y, pd.Series) and y.index.is_integer():
            y, X = _coerce_int_to_range_index(y, X)
        self._fit_forecaster(y, X)
        return self

    def _fit_forecaster(self, y_train, X_train=None):
        """Log used internally in fit."""
        raise NotImplementedError("abstract method")

    def _update(self, y, X=None, update_params=True):
        """Update used internally in update."""
        if update_params or self.is_composite():
            super()._update(y, X, update_params=update_params)
        else:
            if not hasattr(self._fitted_forecaster, "append"):
                warn(
                    f"NotImplementedWarning: {self.__class__.__name__} "
                    f"can not accept new data when update_params=False. "
                    f"Call with update_params=True to refit with new data."
                )
            else:
                # only append unseen data to fitted forecaster
                index_diff = y.index.difference(
                    self._fitted_forecaster.fittedvalues.index
                )
                if index_diff.isin(y.index).all():
                    y = y.loc[index_diff]
                self._fitted_forecaster = self._fitted_forecaster.append(y)

    def _predict(self, fh, X=None):
        """Make forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasters horizon with the steps ahead to to predict.
            Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        y_pred : pd.Series
            Returns series of predicted values.
        """
        # statsmodels requires zero-based indexing starting at the
        # beginning of the training series when passing integers
        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]
        fh_abs = fh.to_absolute(self.cutoff).to_pandas()

        # bug fix for evaluate function as test_plus_train indices are passed
        # statsmodels exog must contain test indices only.
        # For discussion see https://github.com/sktime/sktime/issues/3830
        if X is not None:
            ind_drop = self._X.index
            X = X.loc[~X.index.isin(ind_drop)]
            # Entire range of the forecast horizon is required
            X = X[: fh_abs[-1]]

        if "exog" in inspect.signature(self._forecaster.__init__).parameters.keys():
            y_pred = self._fitted_forecaster.predict(start=start, end=end, exog=X)
        else:
            y_pred = self._fitted_forecaster.predict(start=start, end=end)

        # statsmodels forecasts all periods from start to end of forecasting
        # horizon, but only return given time points in forecasting horizon
        y_pred = y_pred.loc[fh_abs]
        # ensure that name is not added nor removed
        # otherwise this may upset conversion to pd.DataFrame
        y_pred.name = self._y.name
        return y_pred

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        fitted_params = {}
        for name in self._get_fitted_param_names():
            if name in ["aic", "aicc", "bic", "hqic"]:
                fitted_params[name] = getattr(self._fitted_forecaster, name, None)
            else:
                fitted_params[name] = self._fitted_forecaster.params.get(name)
        return fitted_params

    def _get_fitted_param_names(self):
        """Get names of fitted parameters."""
        return self._fitted_param_names


def _coerce_int_to_range_index(y, X=None):
    new_index = pd.RangeIndex(y.index[0], y.index[-1] + 1)
    try:
        np.testing.assert_array_equal(y.index, new_index)
    except AssertionError:
        raise ValueError(
            "Coercion of integer pd.Index to pd.RangeIndex "
            "failed. Please provide `y_train` with a "
            "pd.RangeIndex."
        )
    y.index = new_index
    if X is not None:
        X.index = new_index
    return y, X
