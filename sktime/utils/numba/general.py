# -*- coding: utf-8 -*-
"""General numba utilities."""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def unique_count(X):
    """Numba unique count function for a 1D array."""
    if len(X) > 0:
        X = np.sort(X)
        unique = np.zeros(len(X))
        unique[0] = X[0]
        counts = np.zeros(len(X), dtype=np.int_)
        counts[0] = 1
        unique_count = 0

        for i in X[1:]:
            if i != unique[unique_count]:
                unique_count += 1
                unique[unique_count] = i
                counts[unique_count] = 1
            else:
                counts[unique_count] += 1
        return unique[: unique_count + 1], counts[: unique_count + 1]
    return None, np.zeros(0, dtype=np.int_)


@njit(fastmath=True, cache=True)
def z_normalise_series(X):
    """Numba z-normalisation function for a single time series."""
    std = np.std(X)
    if std > 0:
        X_n = (X - np.mean(X)) / std
    else:
        X_n = np.zeros(len(X))

    return X_n
