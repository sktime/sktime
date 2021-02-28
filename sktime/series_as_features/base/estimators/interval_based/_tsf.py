# -*- coding: utf-8 -*-
"""
    TODO: Currently a work in progress!
    Base Time Series Forest Class.
    An implementation of Deng's Time Series Forest, with minor changes.
"""

__author__ = ["Tony Badnall", "kkoziara", "luiszugasti"]
__all__ = [
    "_transform",
    "_get_intervals",
    "_fit_estimator",
    "_predict_proba_for_estimator",
]

import numpy as np
from sklearn.base import clone

from sktime.utils.slope_and_trend import _slope


# TODO: Determine what names need to be hidden and what names don't
def _transform(X, intervals):
    """Compute the mean, standard deviation and slope for given intervals
    of input data X.

    Args:
        X (Array-like, int or float): Time series data X
        intervals (Array-like, int or float): Time range intervals for series X

    Returns:
        int32 Array: transformed_x containing mean, std_deviation and slope
    """
    n_instances, _ = X.shape
    n_intervals, _ = intervals.shape
    transformed_x = np.empty(shape=(3 * n_intervals, n_instances), dtype=np.float32)
    for j in range(n_intervals):
        X_slice = X[:, intervals[j][0] : intervals[j][1]]
        means = np.mean(X_slice, axis=1)
        std_dev = np.std(X_slice, axis=1)
        slope = _slope(X_slice, axis=1)
        transformed_x[3 * j] = means
        transformed_x[3 * j + 1] = std_dev
        transformed_x[3 * j + 2] = slope

    return transformed_x.T


def _get_intervals(n_intervals, min_interval, series_length, rng):
    """
    Generate random intervals for given parameters.
    """
    intervals = np.zeros((n_intervals, 2), dtype=int)
    for j in range(n_intervals):
        intervals[j][0] = rng.randint(series_length - min_interval)
        length = rng.randint(series_length - intervals[j][0] - 1)
        if length < min_interval:
            length = min_interval
        intervals[j][1] = intervals[j][0] + length
    return intervals


def _fit_estimator(X, y, base_estimator, intervals, random_state=None):
    """
    Fit an estimator - a clone of base_estimator - on input data (X, y)
    transformed using the randomly generated intervals.
    """

    estimator = clone(base_estimator)
    estimator.set_params(random_state=random_state)

    transformed_x = _transform(X, intervals)
    return estimator.fit(transformed_x, y)


def _predict_proba_for_estimator(X, estimator, intervals):
    """
    Find probability estimates for each class for all cases in X using
    given estimator and intervals.
    """
    transformed_x = _transform(X, intervals)
    return estimator.predict_proba(transformed_x)
