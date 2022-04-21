# -*- coding: utf-8 -*-
__author__ = ['chrisholder']

import numpy as np
from numba import njit

from sktime.clustering.metrics.medoids import medoids
from sktime.distances._distance import distance_alignment_path_factory
from tslearn.metrics import dtw_path
from scipy.interpolate import interp1d
from sktime.distances.base import DistanceAlignmentPathCallable


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


def dba(
        X: np.ndarray,
        iterations=30,
        averaging_distance_metric='dtw',
        **kwargs
) -> np.ndarray:
    """Compute the dtw barycenter average of time series.

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n, m, p) where n is number of instances, m
                    is the dimensions and p is the timepoints))
        Time series instances compute average from.
    iterations: int, defaults = 30
        Number iterations for dba to update over.
    distance_metric: str, defaults = 'dtw'
        String that is the distance metric to derive the distance alignment path.
    distance_params: dict, defaults = {}
        Distance parameters to use with distance alignment path.

    Returns
    -------
    np.ndarray (2d array of shape (m, p) where m is the number of dimensions and p is
                the number of time points.)
        The time series that is the computed average series.
    """
    if len(X) <= 1:
        return X

    # center = medoids(X)
    center = _init_avg(X, X.shape[2])
    path_callable = distance_alignment_path_factory(
        X[0],
        X[1],
        metric=averaging_distance_metric,
        **kwargs
    )
    for i in range(iterations):
        center = _dba_update(center, X, path_callable)
    return center


@njit(cache=True)
def _dba_update(
        center: np.ndarray,
        X: np.ndarray,
        path_callable: DistanceAlignmentPathCallable
):
    """Perform an update iteration for dba.

    Parameters
    ----------
    center: np.ndarray (2d array of shape (m, p) where m is the number of dimensions
                        and p is the number of time point)
        Time series that is the current center (or average).
    X : np.ndarray (3d array of shape (n, m, p) where n is number of instances, m
                    is the dimensions and p is the timepoints))
        Time series instances compute average from.
    path_callable: Callable[Union[np.ndarray, np.ndarray], tuple[list[tuple], float]]
        Callable that returns the distance path.

    Returns
    -------
    np.ndarray (2d array of shape (m, p) where m is the number of dimensions and p is
                the number of time points.)
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
