__author__ = ["chrisholder"]

import numpy as np

from sktime.utils.numba.njit import njit


@njit(fastmath=True)
def _dba_update(
    center: np.ndarray, X: np.ndarray, path_callable
) -> tuple[np.ndarray, float]:
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
    sum = np.zeros(X_timepoints)

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
