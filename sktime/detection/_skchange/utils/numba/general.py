"""Numba-optimized functions for various common data manipulation tasks."""

import numpy as np

from sktime.detection._skchange.utils.numba import njit, prange


@njit
def col_repeat(x: np.ndarray, n: int) -> np.ndarray:
    """Repeat each column of a 2D array n times.

    Parameters
    ----------
    x : np.ndarray
        1D array.

    Returns
    -------
    2D array : (x.size, n)-matrix with x in each column
    """
    expanded_x = np.zeros((x.shape[0], n))
    for j in prange(n):
        expanded_x[:, j] = x
    return expanded_x


@njit
def row_repeat(x: np.ndarray, n: int) -> np.ndarray:
    """Repeat each row of a 2D array n times.

    Parameters
    ----------
    x : np.ndarray
        1D array.

    Returns
    -------
    2D array : (n, x.size) matrix with x in each row
    """
    expanded_x = np.zeros((x.shape[0], n))
    for i in prange(n):
        expanded_x[i, :] = x
    return expanded_x


@njit
def where(indicator: np.ndarray) -> list:
    """
    Identify consecutive intervals of True values in the input array.

    Parameters
    ----------
    indicator : np.ndarray
        1D boolean array.

    Returns
    -------
    list of tuples:
        Each tuple represents the start and end indices of consecutive True intervals.
        If there are no True values, an empty list is returned.
    """
    intervals = []
    start, end = None, None
    for i, val in enumerate(indicator):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            intervals.append((start, end))
            start, end = None, None
    if start is not None:
        intervals.append((start, len(indicator)))
    return intervals


@njit
def truncate_below(x: np.ndarray, lower_bound: float) -> np.ndarray:
    """Truncate values below a lower bound.

    Parameters
    ----------
    x : np.ndarray
        2D array.
    lower_bound : float
        Lower bound.

    Returns
    -------
    x : np.ndarray
        2D array with values below lower_bound replaced by lower_bound.
    """
    if x.ndim == 1:
        x[x < lower_bound] = lower_bound
    else:
        p = x.shape[1]
        for j in range(p):
            # Numba doesn't support multidimensional index.
            x[x[:, j] < lower_bound, j] = lower_bound  # To avoid zero division
    return x


@njit
def compute_finite_difference_derivatives(ts: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Compute second-order finite difference derivatives.

    Without assuming uniform sampling, this function computes the second-order
    derivatives of y(t) using a finite difference approximation of the derivative.

    Parameters
    ----------
    ts : np.ndarray
        The sampling points at which to compute the derivatives. Assumed to be sorted.
    ys : np.ndarray
        The values of the function at the sampling points.

    Returns
    -------
    np.ndarray
        The approximated second-order derivatives of y(t) at the sampling points.
    """
    if len(ts) < 3:
        raise ValueError("At least three data points are required.")

    diff_weights = np.zeros((len(ts), len(ts)), dtype=np.float64)
    steps = ts[1:] - ts[:-1]

    # Second-order forward finite difference weights for the first quantile:
    first_steps_sum = steps[0] + steps[1]
    diff_weights[0, 0] = (-2 * steps[0] - steps[1]) / (steps[0] * first_steps_sum)
    diff_weights[0, 1] = first_steps_sum / (steps[0] * steps[1])
    diff_weights[0, 2] = -steps[0] / (steps[1] * first_steps_sum)

    # Central second-order finite difference weights:
    for i in range(1, len(ts) - 1):
        steps_sum = steps[i - 1] + steps[i]

        # For uniform steps, current_weight == 0.
        prev_weight = -steps[i] / (steps_sum * steps[i - 1])
        current_weight = (steps[i] - steps[i - 1]) / (steps[i] * steps[i - 1])
        next_weight = steps[i - 1] / (steps_sum * steps[i])

        diff_weights[i, i - 1] = prev_weight
        diff_weights[i, i] = current_weight
        diff_weights[i, i + 1] = next_weight

    # Second-order backward finite difference weights for the last quantile:
    last_steps_sum = steps[-2] + steps[-1]
    diff_weights[-1, -3] = (steps[-1]) / (steps[-2] * last_steps_sum)
    diff_weights[-1, -2] = (-last_steps_sum) / (steps[-2] * steps[-1])
    diff_weights[-1, -1] = (steps[-1] + last_steps_sum) / (steps[-1] * last_steps_sum)

    derivatives = np.dot(diff_weights, ys)

    return derivatives
