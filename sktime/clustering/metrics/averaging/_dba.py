import numpy as np

from sktime.clustering.metrics.medoids import medoids
from sktime.distances._distance import distance_path_factory


def dba(X: np.ndarray, iterations = 30):
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
    center = medoids(X)
    path_callable = distance_path_factory(X[0], X[1], metric='dtw')
    for i in range(iterations):
        center = _dba_update(center, X, path_callable)



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
    X_size = len(X)
    alignment = np.zeros(X_size)
    for i in range(X_size):
        curr_ts = X[i]
        curr_alignment, _ = path_callable(center, curr_ts)
        for j in range(len(curr_alignment)):
            joe = curr_alignment[j]
            test = X[curr_alignment[j]]
            alignment[j] += X[curr_alignment[j]]

    alignment = alignment/X_size
    return alignment





    pass

