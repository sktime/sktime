# -*- coding: utf-8 -*-
"""Random Interval Spectral Ensemble (RISE)."""

__author__ = ["TonyBagnall"]

import numpy as np

from sktime.utils.numba.njit import jit
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("numba", severity="none"):
    from numba import int64, prange


@jit(parallel=True, cache=True, nopython=True)
def acf(x, max_lag):
    """Autocorrelation function transform.

    currently calculated using standard stats method. We could use inverse of power
    spectrum, especially given we already have found it, worth testing for speed and
    correctness. HOWEVER, for long series, it may not give much benefit, as we do not
    use that many ACF terms.

    Parameters
    ----------
    x : array-like shape = [interval_width]
    max_lag: int
        The number of ACF terms to find.

    Returns
    -------
    y : array-like shape = [max_lag]
    """
    y = np.empty(max_lag)
    length = len(x)
    for lag in prange(1, max_lag + 1):
        # Do it ourselves to avoid zero variance warnings
        lag_length = length - lag
        x1, x2 = x[:-lag], x[lag:]
        s1 = np.sum(x1)
        s2 = np.sum(x2)
        m1 = s1 / lag_length
        m2 = s2 / lag_length
        ss1 = np.sum(x1 * x1)
        ss2 = np.sum(x2 * x2)
        v1 = ss1 - s1 * m1
        v2 = ss2 - s2 * m2
        v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
        if v1_is_zero and v2_is_zero:  # Both zero variance,
            # so must be 100% correlated
            y[lag - 1] = 1
        elif v1_is_zero or v2_is_zero:  # One zero variance
            # the other not
            y[lag - 1] = 0
        else:
            y[lag - 1] = np.sum((x1 - m1) * (x2 - m2)) / np.sqrt(v1 * v2)
        # _x = np.vstack((x[:-lag], x[lag:]))
        # s = np.sum(_x, axis=1)
        # ss = np.sum(_x * _x, axis=1)
        # v = ss - s * s / l
        # zero_variances = v <= 1e-9
        # i = lag - 1
        # if np.all(zero_variances):  # Both zero variance,
        #     # so must be 100% correlated
        #     y[i] = 1
        # elif np.any(zero_variances):  # One zero variance
        #     # the other not
        #     y[i] = 0
        # else:
        #     m = _x - s.reshape(2, 1) / l
        #     y[i] = (m[0] @ m[1]) / np.sqrt(np.prod(v))

    return y


# @jit(parallel=True, cache=True, nopython=True)
def matrix_acf(x, num_cases, max_lag):
    """Autocorrelation function transform.

    Calculated using standard stats method. We could use inverse of power
    spectrum, especially given we already have found it, worth testing for speed and
    correctness. HOWEVER, for long series, it may not give much benefit, as we do not
    use that many ACF terms.

    Parameters
    ----------
    x : array-like shape = [num_cases, interval_width]
    max_lag: int
        The number of ACF terms to find.

    Returns
    -------
    y : array-like shape = [num_cases,max_lag]

    """
    y = np.empty(shape=(num_cases, max_lag))
    length = x.shape[1]
    for lag in prange(1, max_lag + 1):
        # Could just do it ourselves ... TO TEST
        #            s1=np.sum(x[:-lag])/x.shape()[0]
        #            ss1=s1*s1
        #            s2=np.sum(x[lag:])
        #            ss2=s2*s2
        #
        lag_length = length - lag
        x1, x2 = x[:, :-lag], x[:, lag:]
        s1 = np.sum(x1, axis=1)
        s2 = np.sum(x2, axis=1)
        m1 = s1 / lag_length
        m2 = s2 / lag_length
        s12 = np.sum(x1 * x2, axis=1)
        ss1 = np.sum(x1 * x1, axis=1)
        ss2 = np.sum(x2 * x2, axis=1)
        v1 = ss1 - s1 * m1
        v2 = ss2 - s2 * m2
        v12 = s12 - s1 * m2
        v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
        non_zero = ~v1_is_zero & ~v2_is_zero
        # y[:, lag - 1] = np.sum((x1 - m1[:, None]) *
        # (x2 - m2[:, None]), axis=1)
        y[v1_is_zero & v2_is_zero, lag - 1] = 1  # Both zero variance,
        # so must be 100% correlated
        y[v1_is_zero ^ v2_is_zero, lag - 1] = 0  # One zero variance
        # the other not
        var = (v1 * v2)[non_zero]
        y[non_zero, lag - 1] = v12[non_zero] / np.sqrt(var)
    #     # y[lag - 1] = np.corrcoef(x[:, lag:], x[:, -lag])[0][1]
    #     # if np.isnan(y[lag - 1]) or np.isinf(y[lag - 1]):
    #     #     y[lag - 1] = 0
    return y


@jit("int64(int64)", cache=True, nopython=True)
def _round_to_nearest_power_of_two(n):
    return int64(1 << round(np.log2(n)))
