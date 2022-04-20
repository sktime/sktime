# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

from sktime.clustering.metrics.medoids import medoids
from sktime.distances._distance import distance_path_factory
from tslearn.metrics import dtw_path
from scipy.interpolate import interp1d

def _init_avg(_X, barycenter_size):
    X = _X.copy()
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    if X.shape[1] == barycenter_size:
        result = np.nanmean(X, axis=0)
    else:
        X_avg = np.nanmean(X, axis=0)
        xnew = np.linspace(0, 1, barycenter_size)
        f = interp1d(np.linspace(0, 1, X_avg.shape[0]), X_avg,
                     kind="linear", axis=0)
        result = f(xnew)

    result = result.reshape((result.shape[1], result.shape[0]))
    return result



def _dba(X: np.ndarray, iterations=50, distance_metric='dtw'):
    """Compute the dtw barycenter average of time series.

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n_instances, n_dimensions, series_length))
        Time series instances compute average from.
    iterations: int
        Number iterations for dba to update over.

    Returns
    -------
    np.ndarray (2d array of shape (n_dimensions, series_length)
        The time series that is the computed average series.
    """
    if len(X) <= 1:
        return X

    # test = X.copy()
    # test = test.reshape(test.shape[0], test.shape[2], test.shape[1])
    # center = _init_avg(test, X.shape[2])
    # center = center.reshape(center.shape[1], center.shape[0])
    # center = medoids(X)
    center = _init_avg(X, X.shape[2])
    path_callable = distance_path_factory(X[0], X[1], metric=distance_metric)
    for i in range(iterations):
        center = _dba_update(center, X, path_callable)
    return center

@njit(cache=True)
def _dba_update(center: np.ndarray, X: np.ndarray, path_callable):
    """Perform a update iteration for dba.

    Parameters
    ----------
    center: np.ndarray (2d array of shape (series_length, n_dimensions))
        Time series that is the current center (or average).
    X : np.ndarray (3d array of shape (n_instances, series_length, n_dimensions)))
        Time series instances compute average from.

    Returns
    -------
    np.ndarray (2d array of shape (n_dimensions, series_length)
        The time series that is the computed average series.
    """
    X_size, X_dims, X_timepoints = X.shape
    sum = np.zeros((X_timepoints))

    alignment = np.zeros((X_dims, X_timepoints))
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = path_callable(curr_ts, center)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1

    return alignment / sum
