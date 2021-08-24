# -*- coding: utf-8 -*-
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def np_mean(arr):
    """
    Method used to take the mean of a numpy array. np.mean doesn't work in
    njit so this method is aimed to replace it

    Parameters
    ----------
    arr: np.ndarray
        2d array to average

    Returns
    -------
    np.ndarray
        array with averaged values
    """
    avg = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        avg[i] = arr[i, :].mean()
    return avg
