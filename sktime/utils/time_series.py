# -*- coding: utf-8 -*-
__all__ = [
    "_get_n_from_n_timepoints",
    "time_series_slope",
    "fit_trend",
]
__author__ = ["Markus Löning"]

import numpy as np
from sklearn.utils import check_array


def _get_n_from_n_timepoints(n_timepoints, n="sqrt"):
    """
    Get number of intervals from number of time points for various allowed
    input arguments.
    Helpful to compute number of intervals relative to time series length,
    e.g. using floats or functions.

    Parameters
    ----------
    n_timepoints : int
    n : {int, float, str, callable}

    Returns
    -------
    n_intervals_ : int
        Computed number of intervals
    """
    # check input: n_timepoints
    if not np.issubdtype(type(n_timepoints), np.dtype(int).type):
        raise ValueError(
            f"`n_timepoints` must be an integer, but found: " f"{type(n_timepoints)}"
        )
    if not n_timepoints >= 1:
        raise ValueError(f"`n_timepoints` must be >= 1, but found: {n_timepoints}")

    # compute number of splits
    allowed_strings = ["sqrt", "log"]

    # integer
    if np.issubdtype(type(n), np.dtype(int).type):
        if not n <= n_timepoints:
            raise ValueError(
                f"If `n_intervals` is an integer, it must be smaller "
                f"than `n_timepoints`, but found:  `n_intervals`={n} "
                f"and `n_timepoints`={n_timepoints}"
            )
        if n < 1:
            raise ValueError(
                f"If `n_intervals` is an integer, "
                f"`n_intervals` must be >= 1, but found: {n}"
            )
        n_intervals_ = n

    # function
    elif callable(n):
        n_intervals_ = n(n_timepoints)

    # string
    elif isinstance(n, str):
        if n not in allowed_strings:
            raise ValueError(
                f"If `n_intervals` is a string, `n_intervals` must be "
                f"in {allowed_strings}, but found: {n}"
            )
        str_func_map = {"sqrt": np.sqrt, "log": np.log}
        func = str_func_map[n]
        n_intervals_ = func(n_timepoints)

    # float
    elif isinstance(n, float):
        if not (0 < n <= 1):
            raise ValueError(
                f"If `n_intervals` is a float, `n_intervals` must be > 0 "
                f"and <= 1, but found: {n}"
            )
        n_intervals_ = n * n_timepoints

    else:
        raise ValueError(
            f"`n_intervals` must be either one of the allowed string options "
            f"in "
            f"{allowed_strings}, an integer or a float number."
        )

    # make sure n_intervals is an integer and there is at least one interval
    n_intervals_ = np.maximum(1, np.int(n_intervals_))
    return n_intervals_


def fit_trend(x, order=0):
    """Fit linear regression with polynomial terms of given order

        x : array_like, shape=[n_samples, n_obs]
        Time series data, each sample is fitted separately
    order : int
        The polynomial order of the trend, zero is constant (mean), one is
        linear trend, two is quadratic trend, and so on.

    Returns
    -------
    coefs : ndarray, shape=[n_samples, order + 1]
        Fitted coefficients of polynomial order for each sample, one column
        means order zero, two columns mean order 1
        (linear), three columns mean order 2 (quadratic), etc

    See Also
    -------
    add_trend
    remove_trend
    """
    x = check_array(x)

    if order == 0:
        coefs = np.mean(x, axis=1).reshape(-1, 1)

    else:
        n_obs = x.shape[1]
        index = np.arange(n_obs)
        poly_terms = np.vander(index, N=order + 1)

        # linear least squares fitting using numpy's optimised routine,
        # assuming samples in columns
        # coefs = np.linalg.pinv(poly_terms).dot(x.T).T
        coefs, _, _, _ = np.linalg.lstsq(poly_terms, x.T, rcond=None)

        # returning fitted coefficients in expected format with samples in rows
        coefs = coefs.T

    return coefs


def time_series_slope(y, axis=0):
    """Find the slope for each series of y
    Parameters
    ----------
    y: np.ndarray
        Time series
    axis : int, optional (default=0)
        Axis along which to compute slope

    Returns
    ----------
    slope : np.ndarray
        Time series slope
    """
    # Make sure y is always at least 2-dimensional
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Generate time index with correct shape for broadcasting
    shape = np.ones(y.ndim, dtype=np.int)
    shape[axis] *= -1
    x = np.arange(y.shape[axis]).reshape(shape) + 1

    # Precompute mean
    x_mean = x.mean()

    # Compute slope along given axis
    return (np.mean(y * x, axis=axis) - x_mean * np.mean(y, axis=axis)) / (
        (x * x).mean() - x_mean ** 2
    )
