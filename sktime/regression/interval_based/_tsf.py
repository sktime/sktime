# -*- coding: utf-8 -*-
"""
    Time Series Forest Regressor (TSF).
"""

__author__ = ["Tony Bagnall", "kkoziara", "luiszugasti", "kanand77", "Markus Löning"]
__all__ = ["TimeSeriesForestRegressor"]

import numpy as np
from joblib import Parallel
from joblib import delayed
from sklearn.ensemble._forest import ForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sktime.regression.base import BaseRegressor
from sktime.series_as_features.base.estimators.interval_based._tsf import (
    BaseTimeSeriesForest,
)
from sktime.series_as_features.base.estimators.interval_based._tsf import _transform
from sktime.utils.validation.panel import check_X


class TimeSeriesForestRegressor(BaseTimeSeriesForest, ForestRegressor, BaseRegressor):
    """Time series forest regressor.

    A time series forest is an ensemble of decision trees built on random intervals.
     Overview: Input n series length m.
     For each tree
         - sample sqrt(m) intervals,
         - find mean, std and slope for each interval, concatenate to form new
         data set,
         - build decision tree on new data set.
     Ensemble the trees with averaged probability estimates.

     This implementation deviates from the original in minor ways. It samples
     intervals with replacement and does not use the splitting criteria tiny
     refinement described in [1]. This is an intentionally stripped down, non
     configurable version for use as a hive-cote component. For a configurable
     tree based ensemble, see sktime.classifiers.ensemble.TimeSeriesForestClassifier

     Parameters
     ----------
     n_estimators    : int, ensemble size, optional (default = 200)
     min_interval    : int, minimum width of an interval, optional (default
     to 3)
     n_jobs          : int, optional (default=1)
         The number of jobs to run in parallel for both `fit` and `predict`.
         ``-1`` means using all processors.
     random_state    : int, seed for random, optional (default = none)

     Attributes
     ----------
     n_classes    : int
     n_intervals  : int
     classes_    : List of classes for a given problem

     References
     ----------
     .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
     classification and feature extraction",Information Sciences, 239, 2013
     Java implementation
     https://github.com/uea-machine-learning/tsml/blob/master/src/main/
     java/tsml/classifiers/interval_based/TSF.java
     Arxiv version of the paper: https://arxiv.org/abs/1302.2277
    """

    _base_estimator = DecisionTreeRegressor()

    def predict(self, X):
        """Predict

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Panel data

        Returns
        -------
        y_pred : np.array
            Predictions
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        _, series_length = X.shape
        if series_length != self.series_length:
            raise TypeError(
                "The number of time points in the training data does not match "
                "that in the test data."
            )
        y_pred = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict)(X, self.estimators_[i], self.intervals_[i])
            for i in range(self.n_estimators)
        )
        return np.mean(y_pred, axis=0)


def _predict(X, estimator, intervals):
    Xt = _transform(X, intervals)
    return estimator.predict(Xt)
