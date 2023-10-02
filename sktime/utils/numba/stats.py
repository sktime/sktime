"""Numba statistics utilities."""

from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("numba", severity="warning")

import numpy as np  # noqa E402

import sktime.utils.numba.general as general_numba  # noqa E402
from sktime.utils.numba.njit import njit  # noqa E402


@njit(fastmath=True, cache=True)
def mean(X):
    """Numba mean function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    mean : float
        The mean of the input array
    """
    s = 0
    for i in range(X.shape[0]):
        s += X[i]
    return s / X.shape[0]


@njit(fastmath=True, cache=True)
def row_mean(X):
    """Numba mean function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The means for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = mean(X[i])
    return arr


@njit(fastmath=True, cache=True)
def count_mean_crossing(X):
    """Numba count above mean of first order differences for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    mean_crossing_count : float
        The count above mean of first order differences of the input array
    """
    m = mean(X)
    d = general_numba.first_order_differences(X > m)
    count = 0
    for i in range(d.shape[0]):
        if d[i] != 0:
            count += 1
    return count


@njit(fastmath=True, cache=True)
def row_count_mean_crossing(X):
    """Numba count above mean of first order differences for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The count above mean of first order differences for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = count_mean_crossing(X[i])
    return arr


@njit(fastmath=True, cache=True)
def count_above_mean(X):
    """Numba count above mean for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    mean_crossing_count : float
        The count above mean of the input array
    """
    m = mean(X)
    d = X > m
    count = 0
    for i in range(d.shape[0]):
        if d[i] != 0:
            count += 1
    return count


@njit(fastmath=True, cache=True)
def row_count_above_mean(X):
    """Numba count above mean for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The count above mean for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = count_above_mean(X[i])
    return arr


@njit(fastmath=True, cache=True)
def median(X):
    """Numba median function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    median : float
        The median of the input array
    """
    idx = int(X.shape[0] / 2)
    if X.shape[0] % 2 == 1:
        s = np.partition(X, idx)
        return s[idx]
    else:
        s = np.partition(X, [idx - 1, idx])
        return 0.5 * (s[idx - 1] + s[idx])


@njit(fastmath=True, cache=True)
def row_median(X):
    """Numba median function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The medians for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = np.median(X[i])
    return arr


@njit(fastmath=True, cache=True)
def std(X):
    """Numba standard deviation function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    std : float
        The standard deviation of the input array
    """
    m = mean(X)
    s = 0
    for i in range(X.shape[0]):
        s += (X[i] - m) ** 2
    return (s / X.shape[0]) ** 0.5


@njit(fastmath=True, cache=True)
def std2(X, X_mean):
    """Numba standard deviation function for a 1d numpy array with pre-calculated mean.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values
    X_mean : float
        The mean of the input array

    Returns
    -------
    std : float
        The standard deviation of the input array
    """
    s = 0
    for i in range(X.shape[0]):
        s += (X[i] - X_mean) ** 2
    return (s / X.shape[0]) ** 0.5


@njit(fastmath=True, cache=True)
def row_std(X):
    """Numba standard deviation function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The standard deviation for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = std(X[i])
    return arr


@njit(fastmath=True, cache=True)
def numba_min(X):
    """Numba min function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    min : float
        The min of the input array
    """
    m = X[0]
    for i in range(1, X.shape[0]):
        if X[i] < m:
            m = X[i]
    return m


@njit(fastmath=True, cache=True)
def row_numba_min(X):
    """Numba min function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The min for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = numba_min(X[i])
    return arr


@njit(fastmath=True, cache=True)
def numba_max(X):
    """Numba max function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    max : float
        The max of the input array
    """
    m = X[0]
    for i in range(1, X.shape[0]):
        if X[i] > m:
            m = X[i]
    return m


@njit(fastmath=True, cache=True)
def row_numba_max(X):
    """Numba max function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The max for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = numba_max(X[i])
    return arr


@njit(fastmath=True, cache=True)
def slope(X):
    """Numba slope function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    slope : float
        The slope of the input array
    """
    sum_y = 0
    sum_x = 0
    sum_xx = 0
    sum_xy = 0
    for i in range(X.shape[0]):
        sum_y += X[i]
        sum_x += i
        sum_xx += i * i
        sum_xy += X[i] * i
    slope = sum_x * sum_y - X.shape[0] * sum_xy
    denom = sum_x * sum_x - X.shape[0] * sum_xx
    return 0 if denom == 0 else slope / denom


@njit(fastmath=True, cache=True)
def row_slope(X):
    """Numba slope function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The slope for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = slope(X[i])
    return arr


@njit(fastmath=True, cache=True)
def iqr(X):
    """Numba interquartile range function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    iqr : float
        The interquartile range of the input array
    """
    p75, p25 = np.percentile(X, [75, 25])
    return p75 - p25


@njit(fastmath=True, cache=True)
def row_iqr(X):
    """Numba interquartile range function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The interquartile range for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = iqr(X[i])
    return arr


@njit(fastmath=True, cache=True)
def ppv(X):
    """Numba proportion of positive values function for a 1d numpy array.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of values

    Returns
    -------
    ppv : float
        The proportion of positive values range of the input array
    """
    count = 0
    for i in range(X.shape[0]):
        if X[i] > 0:
            count += 1
    return count / X.shape[0]


@njit(fastmath=True, cache=True)
def row_ppv(X):
    """Numba proportion of positive values function for a 2d numpy array.

    Parameters
    ----------
    X : 2d numpy array
        A 2d numpy array of values

    Returns
    -------
    arr : 1d numpy array
        The proportion of positive values for axis 0 of the input array
    """
    arr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        arr[i] = ppv(X[i])
    return arr


@njit(fastmath=True, cache=True)
def fisher_score(X, y):
    """Numba Fisher score function.

    Parameters
    ----------
    X : 1d numpy array
        A 1d numpy array of attribute values
    y : 1d numpy array
        A 1d numpy array of class values

    Returns
    -------
    score : float
        The Fisher score for the given array of attribute values and class values
    """
    unique_labels = np.unique(y)
    mu_feat = mean(X)
    accum_numerator = 0
    accum_denominator = 0

    for k in unique_labels:
        idx_label = np.where(y == k)[0]
        data_sub = X[idx_label]

        mu_feat_label = mean(data_sub)
        sigma_feat_label = max(std2(data_sub, mu_feat_label), 0.000001)

        accum_numerator += idx_label.shape[0] * (mu_feat_label - mu_feat) ** 2
        accum_denominator += idx_label.shape[0] * sigma_feat_label**2

    if accum_denominator == 0:
        return 0
    else:
        return accum_numerator / accum_denominator
