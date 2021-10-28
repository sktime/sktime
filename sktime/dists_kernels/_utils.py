# -*- coding: utf-8 -*-
import numpy as np


def to_numba_timeseries(x: np.ndarray) -> np.ndarray:
    """Convert a timeseries to a valid timeseries for numba use.

    Parameters
    ----------
    x: np.ndarray (1d, 2d or 3d array)
        A timeseries.

    Returns
    -------
    np.ndarray (3d array)
        3d array that is the formatted timeseries.

    Raises
    ------
    ValueError
        If the value provided is not a numpy array
        If the matrix provided is greater than 3 dimensions
    """
    if not isinstance(x, np.ndarray):
        raise ValueError(
            f"The value {x} is an invalid timeseries. To perform a "
            f"distance computation a numpy arrays must be provided."
        )

    _x = np.array(x, copy=True, dtype=np.float)
    num_dims = _x.ndim
    if num_dims == 1:
        shape = _x.shape
        _x = np.reshape(_x, (1, 1, shape[0]))
    elif num_dims == 2:
        shape = _x.shape
        _x = np.reshape(_x, (1, shape[0], shape[1]))
    elif num_dims > 3:
        raise ValueError(
            "The matrix provided has more than 3 dimensions. This is not"
            "supported. Please provide a matrix with less than "
            "3 dimensions"
        )
    return _x
