# -*- coding: utf-8 -*-
"""
    TODO: Currently a work in progress!
    Base Time Series Forest Class.
    An implementation of Deng's Time Series Forest, with minor changes.
"""

__author__ = ["Tony Badnall", "kkoziara", "luiszugasti"]
# __all__ = ["TimeSeriesForest"]

import numpy as np
from sktime.utils.slope_and_trend import _slope


# TODO: Remove this line; helper functions
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
