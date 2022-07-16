# -*- coding: utf-8 -*-
__author__ = ["chrisholder"]

from typing import Tuple

import numpy as np
from numba import njit

from sktime.clustering.metrics.medoids import medoids
from sktime.distances import distance_alignment_path_factory
from sktime.distances.base import DistanceAlignmentPathCallable


def dba(
    X: np.ndarray,
    max_iters: int = 30,
    tol=1e-5,
    averaging_distance_metric: str = "dtw",
    medoids_distance_metric: str = "dtw",
    precomputed_medoids_pairwise_distance: np.ndarray = None,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """Compute the dtw barycenter average of time series.

    This implements the'petitjean' version (orginal) DBA algorithm [1]_.


    Parameters
    ----------
    X : np.ndarray (3d array of shape (n, m, p) where n is number of instances, m
                    is the dimensions and p is the timepoints))
        Time series instances compute average from.
    max_iters: int, defaults = 30
        Maximum number iterations for dba to update over.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the Expectation-Maximization procedure stops.
    averaging_distance_metric: str, defaults = 'dtw'
        String that is the distance metric to derive the distance alignment path.
    medoids_distance_metric: str, defaults = 'euclidean'
        String that is the distance metric to use with medoids
    precomputed_medoids_pairwise_distance: np.ndarray (of shape (len(X), len(X)),
                defulats = None
        Precomputed medoids pairwise.
    verbose: bool, defaults = False
        Boolean that controls the verbosity.

    Returns
    -------
    np.ndarray (2d array of shape (m, p) where m is the number of dimensions and p is
                the number of time points.)
        The time series that is the computed average series.

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    if len(X) <= 1:
        return X

    # center = X.mean(axis=0)
    center = medoids(
        X,
        distance_metric=medoids_distance_metric,
        precomputed_pairwise_distance=precomputed_medoids_pairwise_distance,
    )
    path_callable = distance_alignment_path_factory(
        X[0], X[1], metric=averaging_distance_metric, **kwargs
    )

    cost_prev = np.inf
    for i in range(max_iters):
        center, cost = _dba_update(center, X, path_callable)

        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            break
        else:
            cost_prev = cost

        if verbose is True:
            print(f"[DBA sktime] epoch {i}, cost {cost}")  # noqa: T001
    return center


@njit(fastmath=True)
def _dba_update(
    center: np.ndarray, X: np.ndarray, path_callable: DistanceAlignmentPathCallable
) -> Tuple[np.ndarray, float]:
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
    cost = 0.0
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = path_callable(curr_ts, center)
        for j, k in curr_alignment:
            alignment[:, k] += curr_ts[:, j]
            sum[k] += 1
            cost += np.linalg.norm(curr_ts[:, j] - center[:, k]) ** 2

    return alignment / sum, cost / X_timepoints
