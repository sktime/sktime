#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""copyright: sktime developers, BSD-3-Clause License (see LICENSE file)."""

__author__ = ["Guzal Bulatova", "Markus Löning"]
__all__ = ["ColumnEnsembleForecaster"]

import numpy as np
import pandas as pd

from sklearn.base import clone

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster


class ColumnEnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Forecast each series with separate forecaster.

    Parameters
    ----------
    forecasters : list of tuples (str, estimator, int or str)
        With name as str, estimator as sktime-like estimator,
        index as str or int

    Examples
    --------
    >>> from sktime.forecasting.compose import ColumnEnsembleForecaster
    >>> from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_longley
    >>> _, y = load_longley()
    >>> y = y.drop(columns=["UNEMP", "ARMED", "POP"])
    >>> forecasters = [("trend", PolynomialTrendForecaster(), 0),\
                        ("ses", ExponentialSmoothing(trend='add'), 1)]
    >>> forecaster = ColumnEnsembleForecaster(forecasters=forecasters)
    >>> forecaster.fit(y, fh=[1, 2, 3])
    ColumnEnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _required_parameters = ["forecasters"]
    _tags = {
        "scitype:y": "both",
        "univariate-only": False,
        "y_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, forecasters):
        self.forecasters = forecasters
        super(ColumnEnsembleForecaster, self).__init__(forecasters=forecasters)

    @property
    def _forecasters(self):
        """Make internal list of forecasters.

        The list only contains the name and forecasters, dropping
        the columns. This is for the implementation of get_params
        via _HeterogenousMetaEstimator._get_params which expects
        lists of tuples of len 2.
        """
        return [(name, forecasters) for name, forecasters, _ in self.forecasters]

    @_forecasters.setter
    def _forecasters(self, value):
        self.forecasters = [
            (name, forecasters, columns)
            for ((name, forecasters), (_, _, columns)) in zip(value, self.forecasters)
        ]

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
        names, forecasters, indices = self._check_forecasters()
        if len(set(indices)) != len(indices):
            raise ValueError(
                "One estimator per column required. Found %s unique"
                " estimators" % len(set(indices))
            )
        elif len(set(indices)) != len(y.columns):
            raise ValueError(
                "One estimator per column required. Found %s" % len(indices)
            )
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
        y_pred.index = self.fh.to_absolute(self.cutoff)
        return y_pred

    def get_params(self, deep=True):
        """Get parameters of estimator in `_forecasters`.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("_forecasters", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `_forecasters`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params("_forecasters", **kwargs)
        return self

    def _check_forecasters(self):
        if (
            self.forecasters is None
            or len(self.forecasters) == 0
            or not isinstance(self.forecasters, list)
        ):
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator, int) tuples."
            )
        names, forecasters, indices = zip(*self.forecasters)
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
        return names, forecasters, indices
