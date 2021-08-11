#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements ensemble forecasters that create univariate (optionally weighted)
combination of the predictions from underlying forecasts."""

__author__ = ["mloning", "GuzalBulatova"]
__all__ = ["EnsembleForecaster"]

import numpy as np
import pandas as pd

from scipy.stats import gmean
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
    >>> forecaster = EnsembleForecaster(forecasters=forecasters,\
                                        aggfunc="mean", weights=[1, 10])
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
        if return_pred_int:
            raise NotImplementedError()

        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1)
        y_pred = _aggregate(y=y_pred, aggfunc=self.aggfunc, weights=self.weights)

        return y_pred


def _aggregate(y, aggfunc, weights):
    """Apply aggregation function by row.

    Parameters
    ----------
    y : pd.DataFrame
        Multivariate series to transform.
    aggfunc : str
        Aggregation function used for transformation.
    weights : list of floats
        Weights to apply in aggregation.

    Returns
    -------
    y_agg: pd.Series
        Transformed univariate series.
    """
    if weights is None:
        aggfunc = _check_aggfunc(aggfunc, weighted=False)
        y_agg = aggfunc(y, axis=1)
    else:
        aggfunc = _check_aggfunc(aggfunc, weighted=True)
        y_agg = aggfunc(y, axis=1, weights=np.array(weights))

    return pd.Series(y_agg, index=y.index)


def _check_aggfunc(aggfunc, weighted=False):
    _weighted = "weighted" if weighted else "unweighted"
    valid_aggfuncs = {
        "mean": {"unweighted": np.mean, "weighted": np.average},
        "median": {"unweighted": np.median, "weighted": _weighted_median},
        "min": {"unweighted": np.min, "weighted": _weighted_min},
        "max": {"unweighted": np.max, "weighted": _weighted_max},
        "gmean": {"unweighted": gmean, "weighted": gmean},
    }
    if aggfunc not in valid_aggfuncs.keys():
        raise ValueError("Aggregation function %s not recognized." % aggfunc)
    return valid_aggfuncs[aggfunc][_weighted]


def _weighted_median(y, axis=1, weights=None):
    w_median = np.apply_along_axis(
        func1d=_weighted_percentile,
        axis=axis,
        arr=y.values,
        sample_weight=weights,
        percentile=50,
    )
    return w_median


def _weighted_min(y, axis=1, weights=None):
    w_min = np.apply_along_axis(
        func1d=_weighted_percentile,
        axis=axis,
        arr=y.values,
        sample_weight=weights,
        percentile=0,
    )
    return w_min


def _weighted_max(y, axis=1, weights=None):
    w_max = np.apply_along_axis(
        func1d=_weighted_percentile,
        axis=axis,
        arr=y.values,
        sample_weight=weights,
        percentile=100,
    )
    return w_max
