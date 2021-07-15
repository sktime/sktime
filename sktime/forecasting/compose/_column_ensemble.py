#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Guzal Bulatova", "Markus Löning"]
__all__ = ["ColumnEnsembleForecaster"]

import numpy as np
import pandas as pd

from sklearn.base import clone

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster

from sktime.utils.validation.forecasting import check_X
from sktime.utils.validation.series import check_series


class ColumnEnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Forecast each series and aggregate them into one.

    Parameters
    ----------
    forecasters : list of (str, estimator, index) tuples

    Example
    -------
    >>> from sktime.forecasting.compose import ColumnEnsembleForecaster
    >>> from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> y = pd.DataFrame(np.random.randint(0, 100, size=(100, 2)), columns=list('AB'))
    >>> forecasters = [("trend", PolynomialTrendForecaster(), 0),\
                        ("ses", ExponentialSmoothing(trend='add'), 1)]
    >>> forecaster = ColumnEnsembleForecaster(forecasters=forecasters])
    >>> forecaster.fit(y)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _required_parameters = ["forecasters"]
    _tags = {
        "univariate-only": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, forecasters, aggfunc="mean"):
        self.forecasters = forecasters
        self.aggfunc = aggfunc
        super(ColumnEnsembleForecaster, self).__init__(forecasters=forecasters)

    def fit(self, y, X=None, fh=None):
        """Overrides BaseForecaster's `fit`.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous data
        Returns
        -------
        self : an instance of self.

        State change
        ------------
        stores data in self._X and self._y
        stores fh, if passed
        updates self.cutoff to most recent time in y
        creates fitted model (attributes ending in "_")
        sets is_fitted flag to true
        """
        # if fit is called, fitted state is re-set
        self._is_fitted = False

        self._set_fh(fh)
        y = check_series(y)

        if X is not None:
            X = check_X(X)

        self._X = X
        self._y = y

        self._set_cutoff(y.index[-1])

        self._fit(y=y, X=X, fh=fh)

        # this should happen last
        self._is_fitted = True

        return self

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.
        Returns
        -------
        self : returns an instance of self.
        """
        self.forecasters_ = []

        for (name, forecaster, index) in self.forecasters:
            forecaster_ = clone(forecaster)

            forecaster_.fit(y.iloc[:, index], X, fh)
            self.forecasters_.append((name, forecaster_, index))

        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):

        y_pred = np.zeros((len(fh), len(self.forecasters_)))
        for (_, forecaster, index) in self.forecasters_:
            y_pred[:, index] = forecaster.predict(fh)

        y_pred = pd.DataFrame(data=y_pred)
        return _aggregate(y=y_pred, aggfunc=self.aggfunc)

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("forecasters", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params("forecasters", **kwargs)
        return self


def _aggregate(y, aggfunc, X=None):
    """Apply aggregation function by row.

    Parameters
    ----------
    y : pd.DataFrame
        Multivariate series to transform.
    X : pd.DataFrame, optional (default=None)
        Exogenous data used in transformation.

    Returns
    -------
    column_ensemble: pd.Series
        Transformed univariate series.
    """

    valid_aggfuncs = {"mean": np.mean, "median": np.median}
    if aggfunc not in valid_aggfuncs.keys():
        raise ValueError("Aggregation function %s not recognized." % aggfunc)

    column_ensemble = y.apply(func=valid_aggfuncs[aggfunc], axis=1)

    return pd.Series(column_ensemble, index=y.index)
