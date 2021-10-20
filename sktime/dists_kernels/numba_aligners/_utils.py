# -*- coding: utf-8 -*-
__author__ = ["Chris Holder"]

import numpy as np


def to_distance_timeseries(x):
    """Method to convert timeseries to a valid timeseries for distance.

    Parameters
    ----------
    x: np.ndarray
        Any valid panel or series timeseries

    Returns
    -------
    _x: np.ndarray
        Numpy array that has been converted and formatted
    """
    _x = np.array(x, copy=True, dtype=np.float)
    if _x.ndim < 2:
        _x = np.reshape(x, (-1, 1))
    return _x
