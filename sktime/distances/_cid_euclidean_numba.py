"""Isolated numba imports for CID."""

__author__ = ["jgyasu"]

import numpy as np

from sktime.utils.numba.njit import njit


@njit(cache=True, fastmath=True)
def _numba_cid_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Complexity-Invariant Distance (CID).

    Parameters
    ----------
    x, y : np.ndarray of shape (d, m)
        Input time series.

    Returns
    -------
    float
        CID distance between x and y.
    """
    d, m = x.shape

    ed_sq = 0.0
    cx_sq = 0.0
    cy_sq = 0.0

    for i in range(d):
        for j in range(m - 1):
            # Euclidean distance
            diff = x[i, j] - y[i, j]
            ed_sq += diff * diff

            # Complexity
            dx = x[i, j + 1] - x[i, j]
            dy = y[i, j + 1] - y[i, j]
            cx_sq += dx * dx
            cy_sq += dy * dy

        # Last element for ED
        diff_last = x[i, m - 1] - y[i, m - 1]
        ed_sq += diff_last * diff_last

    # If either complexity is zero, fall back to Euclidean distance
    if cx_sq == 0.0 or cy_sq == 0.0:
        return np.sqrt(ed_sq)

    # Apply complexity correction using squared values
    if cx_sq > cy_sq:
        return np.sqrt(ed_sq * (cx_sq / cy_sq))
    else:
        return np.sqrt(ed_sq * (cy_sq / cx_sq))
