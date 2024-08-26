"""Compute medoids from time series."""

__author__ = ["chrisholder", "TonyBagnall"]

import numpy as np

from sktime.distances import pairwise_distance


def medoids(
    X: np.ndarray,
    precomputed_pairwise_distance: np.ndarray = None,
    distance_metric: str = "dtw",
):
    """Compute the medoids from a panel of time series.

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n_instances, n_dimensions, series_length))
        Time series to compute medoids from.
    precomputed_pairwise_distance: np.ndarray (2d array of shape
        (n_instances, n_instances)), defaults = None
        Precomputed pairwise distance between each time series in X.
    distance_metric: str, defaults = 'dtw'
        String of distance metric to compute.

    Returns
    -------
    np.ndarray (2d array of shape (n_dimensions, series_length)
        The time series that is the medoids.
    """
    if X.shape[0] < 1:
        return X

    if precomputed_pairwise_distance is None:
        precomputed_pairwise_distance = pairwise_distance(X, metric=distance_metric)

    x_size = X.shape[0]
    distance_matrix = np.zeros((x_size, x_size))
    for j in range(x_size):
        for k in range(x_size):
            distance_matrix[j, k] = precomputed_pairwise_distance[j, k]
    return X[np.argmin(sum(distance_matrix))]
