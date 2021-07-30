#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["EnsembleForecaster"]

import numpy as np
import pandas as pd

from sklearn.utils.stats import _weighted_percentile
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster


class EnsembleForecaster(_HeterogenousEnsembleForecaster):
    """Ensemble of forecasters.

    Overview: Input one series of length `n` and EnsembleForecaster performs
    fitting and prediction for each estimator passed in `forecasters`. It then
    applies `aggfunc` aggregation function by row to the predictions dataframe
    and returns final prediction - one series.

    Parameters
    ----------
    forecasters : list of (str, estimator) tuples
        Estimators to apply to the input series.
    n_jobs : int or None, optional, default=None
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    aggfunc : str, {'mean', 'median', 'min', 'max'}, default='mean'
        The function to aggregate prediction from individual forecasters.
    weights : list of floats
        Weights to apply in aggregation.

    Example
    -------
    >>> from sktime.forecasting.compose import EnsembleForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecasters = [("trend", PolynomialTrendForecaster()),\
                        ("naive", NaiveForecaster())]
    >>> forecaster = EnsembleForecaster(forecasters=forecasters, \
                                        aggfunc="min", weights=[1, 10])
    >>> forecaster.fit(y=y, X=None, fh=[1,2,3])
    EnsembleForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    _required_parameters = ["forecasters"]
    _tags = {
        "univariate-only": False,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, forecasters, n_jobs=None, aggfunc="mean", weights=None):
        super(EnsembleForecaster, self).__init__(forecasters=forecasters, n_jobs=n_jobs)
        self.aggfunc = aggfunc
        self.weights = weights

    def _fit(self, y, X=None, fh=None):
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
        names, forecasters = self._check_forecasters()
        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Return the predicted reduction.

        Parameters
        ----------
        fh : int, list or np.array, optional, default=None
        X : pd.DataFrame
        return_pred_int : boolean, optional, default=False
        alpha : fh : float, default=DEFAULT_ALPHA

        Returns
        -------
        y_pred : pd.Series
            Aggregated predictions.
        """
        aggfunc = _check_aggfunc(self.aggfunc)
        if return_pred_int:
            raise NotImplementedError()

        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1)
        y_pred.index = self.fh.to_absolute(self.cutoff)
        return _aggregate(y=y_pred, aggfunc=aggfunc, weights=self.weights)


def _aggregate(y, aggfunc, weights):
    """Apply aggregation function by row.

    Parameters
    ----------
    y : pd.DataFrame
        Multivariate series to transform.
    aggfunc : str
        Aggregation function used for transformation.

    Returns
    -------
    column_ensemble: pd.Series
        Transformed univariate series.
    """
    if aggfunc == np.average:
        y_agg = aggfunc(y, axis=1, weights=weights)
    else:
        y_agg = []
        for _, row in y.iterrows():
            agg = _weighted_percentile(aggfunc(row.to_numpy()), sample_weight=weights)
            y_agg.append(agg)

    return pd.Series(y_agg)


def _check_aggfunc(aggfunc):
    valid_aggfuncs = {
        "mean": np.mean,
        "median": np.median,
        "average": np.average,
        "min": np.min,
        "max": np.max,
    }
    if aggfunc not in valid_aggfuncs.keys():
        raise ValueError("Aggregation function %s not recognized." % aggfunc)
    return valid_aggfuncs[aggfunc]
