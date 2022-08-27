# -*- coding: utf-8 -*-
"""Time Series Forest Regressor (TSF)."""

__author__ = ["Tony Bagnall", "kkoziara", "luiszugasti", "kanand77", "Markus Löning"]
__all__ = ["TimeSeriesForestRegressor"]

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble._forest import ForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sktime.regression.base import BaseRegressor
from sktime.series_as_features.base.estimators.interval_based._tsf import (
    BaseTimeSeriesForest,
    _transform,
)


class TimeSeriesForestRegressor(BaseTimeSeriesForest, ForestRegressor, BaseRegressor):
    """Time series forest regressor.

    A time series forest is an ensemble of decision trees built on random intervals.

    Overview: For input data with n series of length m, for each tree:

    - sample sqrt(m) intervals,
    - find mean, std and slope for each interval, concatenate to form new data set,
    - build decision tree on new data set.

    Ensemble the trees with averaged probability estimates.

    This implementation deviates from the original in minor ways. It samples
    intervals with replacement and does not use the splitting criteria tiny
    refinement described in [1]_. This is an intentionally stripped down, non
    configurable version for use as a HIVE-COTE component.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of estimators.
    min_interval : int, default=3
        Minimum width of an interval.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, default=None

    Attributes
    ----------
    n_classes : int
        Number of classes.
    n_intervals : int
        Number of intervals.
    classes_ : list
        List of classes for a given problem.

    See Also
    --------
    TimeSeriesForestClassifier

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction", Information Sciences, 239, 2013
    .. [2] Java implementation https://github.com/uea-machine-learning/tsml
    .. [3] Arxiv paper: https://arxiv.org/abs/1302.2277
    """

    _tags = {
        "capability:multivariate": False,
        "X_inner_mtype": "numpy3D",
    }

    _base_estimator = DecisionTreeRegressor()

    def __init__(
        self,
        min_interval=3,
        n_estimators=200,
        n_jobs=1,
        random_state=None,
    ):
        super(TimeSeriesForestRegressor, self).__init__(
            min_interval=min_interval,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        BaseRegressor.__init__(self)

    def fit(self, X, y):
        """Override sklearn forest fit with BaseRegressor fit."""
        return BaseRegressor.fit(self, X, y)

    def _fit(self, X, y):
        """Wrap BaseForest._fit.

        This is a temporary measure prior to the BaseRegressor refactor.
        """
        return BaseTimeSeriesForest._fit(self, X, y)

    def predict(self, X):
        """Override sklearn forest predict with BaseRegressor predict."""
        return BaseRegressor.predict(self, X)

    def _predict(self, X):
        """Predict.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Panel data

        Returns
        -------
        np.ndarray
            Predictions.
        """
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
