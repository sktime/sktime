# -*- coding: utf-8 -*-
"""Isolated numba imports for _ddtw."""

__author__ = ["chrisholder", "TonyBagnall"]

import numpy as np

from sktime.utils.numba.njit import njit


@njit(cache=True, fastmath=True)
def average_of_slope(q: np.ndarray) -> np.ndarray:
    r"""Compute the average of a slope between points.

    Computes the average of the slope of the line through the point in question and
    its left neighbour, and the slope of the line through the left neighbour and the
    right neighbour. proposed in [1] for use in this context.

    .. math::
    q'_(i) = \frac{{}(q_{i} - q_{i-1} + ((q_{i+1} - q_{i-1}/2)}{2}

    Where q is the original time series and q' is the derived time series.

    Parameters
    ----------
    q: np.ndarray (of shape (d, m) where d is the dimensions and m is the timepoints.
        A time series.

    Returns
    -------
    np.ndarray (2d array of shape nxm where n is len(q.shape[0]-2) and m is
                len(q.shape[1]))
        Array containing the derivative of q.

    References
    ----------
    .. [1] Keogh E, Pazzani M Derivative dynamic time warping. In: proceedings of 1st
    SIAM International Conference on Data Mining, 2001
    """
    q = q.transpose()
    q2 = 0.25 * q[2:] + 0.5 * q[1:-1] - 0.75 * q[:-2]
    q2 = q2.transpose()
    return q2
