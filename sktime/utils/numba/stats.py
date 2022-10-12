# -*- coding: utf-8 -*-
"""Numba statistic utilities."""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def mean(X):
    """Numba mean function for a single time series."""
    return np.mean(X)


@njit(fastmath=True, cache=True)
def median(X):
    """Numba median function for a single time series."""
    return np.median(X)


@njit(fastmath=True, cache=True)
def std(X):
    """Numba standard deviation function for a single time series."""
    return np.std(X)


@njit(fastmath=True, cache=True)
def numba_min(X):
    """Numba min function for a single time series."""
    return np.min(X)


@njit(fastmath=True, cache=True)
def numba_max(X):
    """Numba max function for a single time series."""
    return np.max(X)


@njit(fastmath=True, cache=True)
def slope(X):
    """Numba slope function for a single time series."""
    sum_y = 0
    sum_x = 0
    sum_xx = 0
    sum_xy = 0

    for i, val in enumerate(X):
        sum_y += val
        sum_x += i
        sum_xx += i * i
        sum_xy += val * i

    slope = sum_xy - (sum_x * sum_y) / len(X)
    denom = sum_xx - (sum_x * sum_x) / len(X)

    return 0 if denom == 0 else slope / denom


@njit(fastmath=True, cache=True)
def iqr(X):
    """Numba interquartile range function for a single time series."""
    sorted = X.copy()
    sorted.sort()

    return np.percentile(sorted, 75) - np.percentile(sorted, 25)
