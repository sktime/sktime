"""Utility functions to calculate window length from tolerance coefficients."""

# found module but no type hints or library stubs
import numpy as np

from sktime.libs.fracdiff import fdiff_coef


def window_from_tol_coef(n: float, tol_coef: float, max_window: int = 2**12) -> int:
    """
    Return length of window determined from tolerance to memory loss.

    Tolerance of smallness of coefficient to determine the length of window.
    That is, `window_` is chosen as the minimum integer that makes the
    absolute value of the `window`-th fracdiff coefficient is smaller than
    `tol_coef`.

    Parameters
    ----------
    - n : int
        Order of fractional differentiation.
    - tol_coef : float in range (0, 1)
        ...

    Notes
    -----
    The window for small `d` or `tol_(memory|coef)` can become extremely large.
    For instance, window grows with the order of `tol_coef ** (-1 / d)`.

    Returns
    -------
    window : int
        Length of window

    Examples
    --------
    >>> from sktime.libs.fracdiff.sklearn.tol import window_from_tol_coef
    >>> from sktime.libs.fracdiff import fdiff_coef
    >>> window_from_tol_coef(0.5, 0.1)
    4
    >>> fdiff_coef(0.5, 3)[-1]
    -0.125
    >>> fdiff_coef(0.5, 4)[-1]
    -0.0625
    """
    coef = np.abs(fdiff_coef(n, max_window))
    return int(np.searchsorted(-coef, -tol_coef) + 1)  # index -> length


def window_from_tol_memory(n: float, tol_memory: float, max_window: int = 2**12) -> int:
    """
    Return length of window determined from tolerance to memory loss.

    Minimum length of window that makes the absolute value of the sum of fracdiff
    coefficients from `window_ + 1`-th term is smaller than `tol_memory`.
    If `window` is not None, ignored.

    Notes
    -----
    The window for small `d` or `tol_(memory|coef)` can become extremely large.
    For instance, window grows with the order of `tol_coef ** (-1 / d)`.

    Parameters
    ----------
    - n : int
        Order of fractional differentiation.
    - tol_memory : float in range (0, 1)
        Tolerance of lost memory.

    Returns
    -------
    window : int
        Length of window

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.libs.fracdiff import fdiff_coef
    >>> from sktime.libs.fracdiff.sklearn.tol import window_from_tol_memory
    >>>
    >>> window_from_tol_memory(0.5, 0.2)
    9
    >>> np.sum(fdiff_coef(0.5, 10000)[9:])
    -0.19073...
    >>> np.sum(fdiff_coef(0.5, 10000)[8:])
    -0.20383...
    """
    lost_memory = np.abs(np.cumsum(fdiff_coef(n, max_window)))
    return int(np.searchsorted(-lost_memory, -tol_memory) + 1)  # index -> length
