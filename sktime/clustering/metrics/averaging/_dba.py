import numpy as np

from sktime.clustering.metrics.medoids import medoids


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
    for i in range(iterations):
        center = _dba_update(center, X)



def _dba_update(center: np.ndarray, X: np.ndarray):
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
    pass

