"""General numba utilities."""

import numpy as np

import sktime.utils.numba.stats as stats  # noqa E402
from sktime.utils.numba.njit import njit  # noqa E402


@njit(fastmath=True, cache=True)
def unique_count(X):
    """Numba unique value count function for a 1d numpy array.

    np.unique() is supported by numba, but the return_counts parameter is not.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    unique : 1d numpy array
        The unique values in X
    counts : 1d numpy array
        The occurrence count for each unique value in X
    """
    if X.shape[0] > 0:
        X = np.sort(X)
        unique = np.zeros(X.shape[0])
        unique[0] = X[0]
        counts = np.zeros(X.shape[0], dtype=np.int_)
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
def first_order_differences(X):
    """Numba first order differences function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    arr : 1d numpy array of size (X.shape[0] - 1)
        The first order differences of X
    """
    return X[1:] - X[:-1]


@njit(fastmath=True, cache=True)
def row_first_order_differences(X):
    """Numba first order differences function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 2d numpy array of shape (X.shape[0], X.shape[1] - 1)
        The first order differences for axis 0 of the input array
    """
    return X[:, 1:] - X[:, :-1]


@njit(fastmath=True, cache=True)
def z_normalise_series(X):
    """Numba series normalization function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The normalised series
    """
    s = stats.std(X)
    if s > 0:
        arr = (X - stats.mean(X)) / s
    else:
        arr = X - stats.mean(X)
    return arr


@njit(fastmath=True, cache=True)
def z_normalise_series_2d(X):
    """Numba series normalization function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 2d numpy array
        The normalised series
    """
    arr = np.zeros(X.shape)
    for i in range(X.shape[0]):
        arr[i] = z_normalise_series(X[i])
    return arr


@njit(fastmath=True, cache=True)
def z_normalise_series_3d(X):
    """Numba series normalization function for a 3d numpy array.

    Parameters
    ----------
    X : 3d numpy array
        A 3d numpy array of values

    Returns
    -------
    arr : 3d numpy array
        The normalised series
    """
    arr = np.zeros(X.shape)
    for i in range(X.shape[0]):
        arr[i] = z_normalise_series_2d(X[i])
    return arr
