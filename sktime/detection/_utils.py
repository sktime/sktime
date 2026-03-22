# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Ported from skchange (BSD-3-Clause), original authors: Tveten, johannvk
"""Utility and validation functions for the detection module."""

__author__ = ["Tveten", "johannvk"]

import numbers

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# --- Data validation utilities ---


def check_data(
    X,
    min_length,
    min_length_name="min_length",
    allow_missing_values=False,
):
    """Check if input data is valid and coerce to pd.DataFrame.

    Parameters
    ----------
    X : pd.DataFrame, pd.Series, np.ndarray
        Input data to check.
    min_length : int
        Minimum number of samples in X.
    min_length_name : str, optional (default="min_length")
        Name of min_length parameter for error messages.
    allow_missing_values : bool, optional (default=False)
        Whether to allow missing values in X.

    Returns
    -------
    X : pd.DataFrame
        Input data in pd.DataFrame format.
    """
    X = pd.DataFrame(X)

    if not allow_missing_values and X.isna().any(axis=None):
        raise ValueError(
            f"X cannot contain missing values: X.isna().sum()={X.isna().sum()}."
        )

    n = X.shape[0]
    if n < min_length:
        raise ValueError(
            f"X must have at least {min_length_name}={min_length} samples"
            + f" (X.shape[0]={n})"
        )

    return X


def as_2d_array(X, vector_as_column=True, dtype=None):
    """Convert an array-like object to a 2D numpy array.

    Parameters
    ----------
    X : ArrayLike
        Array-like object.
    vector_as_column : bool, optional (default=True)
        If True, a 1D array is reshaped to a column vector.
        If False, a 1D array is reshaped to a row vector.
    dtype : data-type, optional
        Desired data-type for the array.

    Returns
    -------
    X : np.ndarray
        2D numpy array.
    """
    X = np.asarray(X, dtype=dtype)
    if X.ndim == 1:
        X = X.reshape(-1, 1) if vector_as_column else X.reshape(1, -1)
    elif X.ndim > 2:
        raise ValueError("X must be at most 2-dimensional.")
    return X


# --- Cut validation utilities ---


def check_cuts_array(cuts, n_samples, min_size=None, last_dim_size=2):
    """Check array type cuts.

    Parameters
    ----------
    cuts : np.ndarray
        Array of cuts to check.
    n_samples : int
        Number of samples in the data.
    min_size : int, optional (default=1)
        Minimum size of the intervals obtained by the cuts.
    last_dim_size : int, optional (default=2)
        Size of the last dimension.

    Returns
    -------
    cuts : np.ndarray
        The unmodified input cuts array.

    Raises
    ------
    ValueError
        If the cuts does not meet the requirements.
    """
    if min_size is None:
        min_size = 1

    if cuts.ndim != 2:
        raise ValueError("The cuts must be a 2D array.")

    if not np.issubdtype(cuts.dtype, np.integer):
        raise ValueError("The cuts must be of integer type.")

    if cuts.shape[-1] != last_dim_size:
        raise ValueError(
            "The cuts must be specified as an array with length "
            f"{last_dim_size} in the last dimension."
        )

    if not np.all(cuts >= 0) or not np.all(cuts <= n_samples):
        raise ValueError(
            "All cuts must be non-negative, and less than "
            f"or equal to the number of samples=({n_samples})."
        )

    interval_sizes = np.diff(cuts, axis=1)
    if not np.all(interval_sizes >= min_size):
        min_interval_size = np.min(interval_sizes)
        raise ValueError(
            "All rows in `cuts` must be strictly increasing and each entry must"
            f" be more than min_size={min_size} apart."
            f" Found a minimum interval size of {min_interval_size}."
        )
    return cuts


# --- Parameter validation utilities ---


def check_larger_than(min_value, value, name, allow_none=False):
    """Check if ``value`` is larger than or equal to ``min_value``.

    Parameters
    ----------
    min_value : int or float
        Minimum allowed value.
    value : int or float
        Value to check.
    name : str
        Name of the parameter for error messages.
    allow_none : bool, optional (default=False)
        Whether to allow None values.

    Returns
    -------
    value : int or float
        Input value.
    """
    if not allow_none and value is None:
        raise ValueError(f"{name} cannot be None.")
    if value is not None and value < min_value:
        raise ValueError(f"{name} must be at least {min_value} ({name}={value}).")
    return value


def check_smaller_than(max_value, value, name, allow_none=False):
    """Check if ``value`` is smaller than or equal to ``max_value``.

    Parameters
    ----------
    max_value : int or float
        Maximum allowed value.
    value : int or float
        Value to check.
    name : str
        Name of the parameter for error messages.
    allow_none : bool, optional (default=False)
        Whether to allow None values.

    Returns
    -------
    value : int or float
        Input value.
    """
    if not allow_none and value is None:
        raise ValueError(f"{name} cannot be None.")
    if value is not None and value > max_value:
        raise ValueError(f"{name} must be at most {max_value} ({name}={value}).")
    return value


def check_in_interval(interval, value, name, allow_none=False):
    """Check if ``value`` is within ``interval``.

    Parameters
    ----------
    interval : pd.Interval
        Interval to check.
    value : int or float
        Value to check.
    name : str
        Name of the parameter for error messages.
    allow_none : bool, optional (default=False)
        Whether to allow None values.

    Returns
    -------
    value : int or float
        Input value.
    """
    if not allow_none and value is None:
        raise ValueError(f"{name} cannot be None.")
    if value is not None and value not in interval:
        raise ValueError(f"{name} must be in {interval} ({name}={value}).")
    return value


# --- Cost parameter validation utilities ---

MeanType = ArrayLike | numbers.Number
VarType = ArrayLike | numbers.Number
CovType = ArrayLike | numbers.Number


def check_mean(mean, X):
    """Check if the fixed mean parameter is valid.

    Parameters
    ----------
    mean : np.ndarray or numbers.Number
        Fixed mean for cost calculation.
    X : np.ndarray
        2D input data.

    Returns
    -------
    mean : np.ndarray
        Fixed mean for cost calculation.
    """
    mean = np.array([mean]) if isinstance(mean, numbers.Number) else np.asarray(mean)
    if len(mean) != 1 and len(mean) != X.shape[1]:
        raise ValueError(f"mean must have length 1 or X.shape[1], got {len(mean)}.")
    return mean


def check_var(var, X):
    """Check if the fixed variance parameter is valid.

    Parameters
    ----------
    var : np.ndarray or numbers.Number
        Fixed variance for cost calculation.
    X : np.ndarray
        2D input data.

    Returns
    -------
    var : np.ndarray
        Fixed variance for cost calculation.
    """
    var = np.array([var]) if isinstance(var, numbers.Number) else np.asarray(var)
    if len(var) != 1 and len(var) != X.shape[1]:
        raise ValueError(f"var must have length 1 or X.shape[1], got {len(var)}.")
    if np.any(var <= 0):
        raise ValueError("var must be positive.")
    return var


def check_cov(cov, X, force_float=False):
    """Check if the fixed covariance matrix parameter is valid.

    Parameters
    ----------
    cov : np.ndarray or numbers.Number
        Fixed covariance matrix for cost calculation.
    X : np.ndarray
        2D input data.
    force_float : bool, default=False
        If True, force the covariance matrix to be of floating point data type.

    Returns
    -------
    cov : np.ndarray
        Fixed covariance matrix for cost calculation.
    """
    p = X.shape[1]
    cov = cov * np.eye(p) if isinstance(cov, numbers.Number) else np.asarray(cov)

    if cov.ndim != 2:
        raise ValueError(f"cov must have 2 dimensions, got {cov.ndim}.")
    if cov.shape[0] != p or cov.shape[1] != p:
        raise ValueError(
            f"cov must have shape (X.shape[1], X.shape[1]), got {cov.shape}."
        )
    if not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError("covariance matrix must be positive definite.")
    if force_float:
        cov = cov.astype(float)
    return cov


def check_non_negative_parameter(scale, X):
    """Check if the fixed scale parameter is valid and non-negative.

    Parameters
    ----------
    scale : np.ndarray or numbers.Number
        Fixed scale for cost calculation.
    X : np.ndarray
        2D input data.

    Returns
    -------
    scale : np.ndarray
        Fixed scale for cost calculation.
    """
    scale = (
        np.array([scale]) if isinstance(scale, numbers.Number) else np.asarray(scale)
    )
    if len(scale) != 1 and len(scale) != X.shape[1]:
        raise ValueError(
            f"Parameter must have length 1 or X.shape[1], got {len(scale)}."
        )
    if np.any(scale <= 0.0):
        raise ValueError("Parameter must be positive.")
    return scale


# --- Penalty validation utilities ---


def check_penalty(penalty, arg_name, caller_name, allow_none=True):
    """Check if the given penalty is valid.

    Parameters
    ----------
    penalty : np.ndarray, float, or None
        The penalty to check.
    arg_name : str
        The name of the argument for error messages.
    caller_name : str
        The name of the caller for error messages.
    allow_none : bool, default=True
        If True, the penalty can be None.
    """
    import copy

    penalty = copy.deepcopy(penalty)

    if not allow_none and penalty is None:
        raise ValueError(f"`{arg_name}` cannot be None in {caller_name}")
    if penalty is None:
        return None

    penalty = np.atleast_1d(np.asarray([penalty]).squeeze())
    if penalty.ndim != 1:
        raise ValueError(
            f"`{arg_name}` must be a 1D array in {caller_name}."
            f" Got {penalty.ndim}D array."
        )
    if penalty.size < 1:
        raise ValueError(
            f"`{arg_name}` must have at least one element in {caller_name}"
        )
    if not np.all(penalty >= 0.0):
        raise ValueError(f"`{arg_name}` must be non-negative in {caller_name}")
    if not np.all(np.diff(penalty) >= 0):
        raise ValueError(f"`{arg_name}` must be non-decreasing in {caller_name}")


def check_penalty_against_data(penalty, X, caller_name):
    """Check if the given penalty is valid against the data.

    Parameters
    ----------
    penalty : np.ndarray or float
        The penalty to check.
    X : np.ndarray
        The data to check against.
    caller_name : str
        The name of the caller for error messages.
    """
    import copy

    penalty = copy.deepcopy(penalty)
    penalty = np.asarray([penalty]).flatten()

    if penalty.size != 1 and penalty.size != X.shape[1]:
        raise ValueError(
            f"`penalty` must be a single value or an array of size"
            f" `X.shape[1]={X.shape[1]}` in {caller_name}."
            f" Got `penalty.size={penalty.size}`."
        )


# --- Interval scorer validation utilities ---


def check_interval_scorer(
    scorer, arg_name, caller_name, required_tasks=None, allow_penalised=True
):
    """Check if the given scorer is a valid interval scorer.

    Parameters
    ----------
    scorer : BaseIntervalScorer
        The scorer to check.
    arg_name : str
        The name of the argument for error messages.
    caller_name : str
        The name of the caller for error messages.
    required_tasks : list or None, optional
        If provided, the scorer's task must be in this list.
    allow_penalised : bool, default=True
        If False, penalised scorers are not allowed.
    """
    from sktime.detection.base._base_interval_scorer import BaseIntervalScorer

    if not isinstance(scorer, BaseIntervalScorer):
        raise ValueError(
            f"`{arg_name}` must be a BaseIntervalScorer. Got {type(scorer)}."
        )
    task = scorer.get_tag("task")
    if required_tasks and task not in required_tasks:
        _required_tasks = [f'"{t}"' for t in required_tasks]
        tasks_str = (
            ", ".join(_required_tasks[:-1]) + " or " + _required_tasks[-1]
            if len(_required_tasks) > 1
            else _required_tasks[0]
        )
        raise ValueError(
            f"{caller_name} requires `{arg_name}` to have task {tasks_str}"
            f" ({arg_name}.get_tag('task') in {required_tasks}). "
            f'Got {scorer.__class__.__name__}, which has task "{task}".'
        )
    if not allow_penalised and scorer.get_tag("is_penalised"):
        raise ValueError(f"`{arg_name}` cannot be a penalised score.")


# --- Array helpers ---


def where_positive(scores):
    """Find contiguous intervals where 1D scores are above zero.

    Parameters
    ----------
    scores : np.ndarray
        1D array of scores.

    Returns
    -------
    list of tuple(int, int)
        Each ``(start, end)`` pair gives the start (inclusive) and end
        (exclusive) of a contiguous interval where ``scores > 0``.
    """
    mask = scores > 0
    if not np.any(mask):
        return []
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [len(scores)]))
    return list(zip(starts.tolist(), ends.tolist()))


# --- Soft numba JIT decorator ---
# Never import numba at module level. This decorator falls back to an identity
# decorator if numba is not installed, matching skchange's behavior.


def _soft_njit(func=None, **kwargs):
    """Apply numba.njit if available, otherwise return function unchanged.

    This avoids module-level numba imports which crash CI environments
    without numba installed.
    """
    try:
        from numba import njit as _njit

        if func is not None:
            return _njit(func, **kwargs)
        else:
            return _njit(**kwargs)
    except ImportError:
        if func is not None:
            return func
        else:

            def decorator(f):
                return f

            return decorator


# --- Numba-accelerated statistical helpers ---
# These are ported as pure numpy with optional numba JIT.
# The numba decorator is applied lazily to avoid import-time crashes.

_col_cumsum_compiled = None


def col_cumsum(x, init_zero=False):
    """Calculate the cumulative sum of each column in a 2D array.

    Parameters
    ----------
    x : np.ndarray
        2D array.
    init_zero : bool
        Whether to prepend a row of zeros.

    Returns
    -------
    np.ndarray
        Cumulative sums. If init_zero, output has one more row than input.
    """
    global _col_cumsum_compiled
    if _col_cumsum_compiled is not None:
        return _col_cumsum_compiled(x, init_zero)

    # Try to use numba-accelerated version
    try:
        from numba import njit as _njit

        @_njit
        def _col_cumsum_numba(x, init_zero):
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

        _col_cumsum_compiled = _col_cumsum_numba
        return _col_cumsum_compiled(x, init_zero)
    except ImportError:
        pass

    # Pure numpy fallback
    n = x.shape[0]
    p = x.shape[1]
    if init_zero:
        sums = np.zeros((n + 1, p))
        sums[1:, :] = np.cumsum(x, axis=0)
    else:
        sums = np.cumsum(x, axis=0)
    return sums


_truncate_below_compiled = None


def truncate_below(x, lower_bound):
    """Truncate values below a lower bound.

    Parameters
    ----------
    x : np.ndarray
        Array (1D or 2D).
    lower_bound : float
        Lower bound.

    Returns
    -------
    x : np.ndarray
        Array with values below lower_bound replaced by lower_bound.
    """
    global _truncate_below_compiled
    if _truncate_below_compiled is not None:
        return _truncate_below_compiled(x, lower_bound)

    try:
        from numba import njit as _njit

        @_njit
        def _truncate_numba(x, lower_bound):
            if x.ndim == 1:
                x[x < lower_bound] = lower_bound
            else:
                p = x.shape[1]
                for j in range(p):
                    x[x[:, j] < lower_bound, j] = lower_bound
            return x

        _truncate_below_compiled = _truncate_numba
        return _truncate_below_compiled(x, lower_bound)
    except ImportError:
        pass

    # Pure numpy fallback
    return np.maximum(x, lower_bound)


def col_median(X, output_array=None):
    """Compute the column-wise median of a 2D array.

    Parameters
    ----------
    X : np.ndarray
        2D array.
    output_array : np.ndarray, optional
        If provided, results are written into this array.

    Returns
    -------
    medians : np.ndarray
        Column-wise medians.
    """
    result = np.median(X, axis=0)
    if output_array is not None:
        output_array[:] = result
        return output_array
    return result


def log_det_covariance(X):
    """Compute the log determinant of the sample covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        2D array with shape (n_samples, n_variables).

    Returns
    -------
    log_det : float
        Log determinant of the sample covariance matrix.
        Returns ``np.nan`` if the covariance matrix is not positive definite.
    """
    n = X.shape[0]
    mean = np.mean(X, axis=0)
    centered = X - mean
    cov = (centered.T @ centered) / n
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.nan
    return logdet


def compute_finite_difference_derivatives(ts, ys):
    """Compute finite difference derivatives of ys with respect to ts.

    Uses central differences for interior points, forward/backward at edges.

    Parameters
    ----------
    ts : np.ndarray
        1D array of independent variable values.
    ys : np.ndarray
        1D array of dependent variable values.

    Returns
    -------
    derivatives : np.ndarray
        1D array of finite difference derivatives.
    """
    n = len(ts)
    derivs = np.empty(n)
    if n == 1:
        derivs[0] = 0.0
        return derivs
    derivs[0] = (ys[1] - ys[0]) / (ts[1] - ts[0])
    derivs[-1] = (ys[-1] - ys[-2]) / (ts[-1] - ts[-2])
    for i in range(1, n - 1):
        derivs[i] = (ys[i + 1] - ys[i - 1]) / (ts[i + 1] - ts[i - 1])
    return derivs
