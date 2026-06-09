"""Numba-optimized functions for calculating various statistics."""

import numpy as np

from sktime.detection._skchange.utils.numba import njit


@njit
def col_cumsum(x: np.ndarray, init_zero: bool = False) -> np.ndarray:
    """Calculate the cumulative sum of each column in a 2D array.

    Parameters
    ----------
    x : np.ndarray
        2D array.
    init_zero : bool
        Whether to let the first row be a row of zeros before the summing is
        started or not.

    Returns
    -------
    np.ndarray : Cumulative sums. If init_zero, the output contains one more
        row compared to the input x.

    """
    n = x.shape[0]
    p = x.shape[1]
    if init_zero:
        sums = np.zeros((n + 1, p))
        start = 1
    else:
        sums = np.zeros((n, p))
        start = 0

    for j in range(p):
        sums[start:, j] = np.cumsum(x[:, j])

    return sums


@njit
def col_median(x: np.ndarray, output_array: np.ndarray | None = None) -> np.ndarray:
    """Calculate the median of each column in a 2D array.

    Parameters
    ----------
    x : np.ndarray
        2D array.
    output_array : np.ndarray
        Array to store the computed medians. If None, a new array is created.

    Returns
    -------
    np.ndarray : Medians.

    """
    p = x.shape[1]
    if output_array is None:
        output_array = np.zeros(p)
    else:
        assert len(output_array) == p

    for j in range(p):
        output_array[j] = np.median(x[:, j])

    return output_array


@njit
def log_det_covariance(X: np.ndarray) -> float:
    """Compute log determinant of the covariance matrix of a data matrix.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n, p) where n is the number of samples and p is the number of
        variables.

    Returns
    -------
    log_abs_det : float
        The log of the absolute value of the determinant of the covariance matrix.
        Returns np.nan if the covariance matrix is not positive definite.

    """
    p = X.shape[1]
    cov = np.cov(X, rowvar=False, ddof=0).reshape(p, p)
    det_sign, log_abs_det = np.linalg.slogdet(cov)

    if det_sign <= 0:
        return np.nan
    else:
        return log_abs_det


@njit
def log_gamma(x: float) -> float:
    """Compute the log of the gamma function.

    Uses the Stirling's approximation for the gamma function.
    The accuracy of the approximation is not good for small x.
    Expect to never evaluate the log_gamma function for x <= 1.0,
    when used to calculate the log-likelihood of a multivariate t-distribution.
    Source: https://en.wikipedia.org/wiki/Gamma_function#Log-gamma_function

    Parameters
    ----------
    x : float
        Positive real argument for the gamma function.

    Returns
    -------
    log_gamma : float
        The log of the gamma function evaluated at x.
    """
    if x <= 1.0e-2:
        return np.nan

    x_cubed = x * x * x
    log_gamma = (
        (x - 0.5) * np.log(x)
        - x
        + 0.5 * np.log(2.0 * np.pi)
        + 1.0 / (12.0 * x)
        - 1.0 / (360.0 * x_cubed)
        + 1.0 / (1260.0 * x_cubed * x * x)
    )

    return log_gamma


@njit
def digamma(x: float) -> float:
    """Approximate the digamma function.

    Use the asymptotic expansion for the digamma function on the real domain,
    by first moving the argument above 5.0 before
    applying the first three terms of its asymptotic expansion.
    The accuracy of the approximation is not good for small x,
    but we do not expect to evaluate the digamma function at
    values x <= 0.5 in the context of evaluating the log-likelihood
    of a multivariate t-distribution.

    Source: https://en.wikipedia.org/wiki/Digamma_function#Asymptotic_expansion

    Parameters
    ----------
    x : float
        Positive real argument for the digamma function.

    Returns
    -------
    digamma : float
        The digamma function evaluated at x.
    """
    if x <= 1.0e-2:
        return np.nan

    result = 0.0
    while x <= 5.0:
        result -= 1.0 / x
        x += 1.0
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += np.log(x) - 0.5 * inv_x - inv_x2 * (1.0 / 12.0 - inv_x2 / 120.0)
    return result


@njit
def trigamma(x: float) -> float:
    """
    Approximate the trigamma function on the real positive domain.

    Uses the asymptotic expansion for the trigamma function on the real domain,
    by first moving the argument above 5.0 before
    applying the first four terms of its asymptotic expansion.
    The accuracy of the approximation is not good for small x,
    but we do not expect to evaluate the trigamma function at
    values x <= 0.5 in the context of evaluating the log-likelihood
    of a multivariate t-distribution.

    Source: https://en.wikipedia.org/wiki/Trigamma_function

    Parameters
    ----------
    x : float
        Positive real argument for the trigamma function.

    Returns
    -------
    trigamma : float
        The trigamma function evaluated at x.
    """
    if x <= 1.0e-2:
        return np.nan

    result = 0.0
    while x <= 5.0:
        result += 1.0 / (x * x)
        x += 1.0
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += (
        (1.0 / x)
        + 0.5 * inv_x2
        + (1.0 / 6.0) * inv_x2 * inv_x
        + (1.0 / 30.0) * inv_x2 * inv_x2 * inv_x
    )
    return result


@njit
def kurtosis(centered_samples: np.ndarray, fisher=True) -> float:
    """Compute the kurtosis of a set of samples.

    Parameters
    ----------
    centered_samples : np.ndarray
        Centered samples, shape (n_samples, n_features).
    fisher : bool (default=True)
        Whether to apply the Fisher correction or not,
        subtracting 3.0, resulting in a Kurtosis of
        zero for the normal distribution.

    Returns
    -------
    per_dim_kurtosis : np.ndarray
        Kurtosis values for each feature.
    """
    sample_dim = centered_samples.shape[1]
    per_dim_squared_variance = np.zeros(sample_dim)
    for i in range(sample_dim):
        per_dim_squared_variance[i] = np.var(centered_samples[:, i]) ** 2

    per_dim_fourth_moment = np.zeros(sample_dim)
    for i in range(sample_dim):
        zero_mean_samples = centered_samples[:, i] - np.mean(centered_samples[:, i])
        per_dim_fourth_moment[i] = np.mean(zero_mean_samples**4)

    per_dim_kurtosis = per_dim_fourth_moment / per_dim_squared_variance
    if fisher:
        per_dim_kurtosis -= 3.0

    return per_dim_kurtosis
