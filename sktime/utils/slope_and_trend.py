"""Slope and trend utilities."""
__all__ = [
    "_slope",
    "_fit_trend",
]
__author__ = ["mloning"]

import numpy as np
from sklearn.utils import check_array


def _fit_trend(x, order=0):
    """Fit linear regression with polynomial terms of given order.

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
    --------
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


def _slope(y, axis=0):
    """Find the slope for each series of y.

    Parameters
    ----------
    y: np.ndarray
        Time series
    axis : int, optional (default=0)
        Axis along which to compute slope

    Returns
    -------
    slope : np.ndarray
        Time series slope
    """
    # Make sure y is always at least 2-dimensional
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Generate time index with correct shape for broadcasting
    shape = np.ones(y.ndim, dtype=int)
    shape[axis] *= -1
    x = np.arange(y.shape[axis]).reshape(shape) + 1

    # Precompute mean
    x_mean = x.mean()

    # Compute slope along given axis
    return (np.mean(y * x, axis=axis) - x_mean * np.mean(y, axis=axis)) / (
        (x * x).mean() - x_mean**2
    )
