"""Fractional differentiation core implementation."""

from functools import partial
from typing import Optional

import numpy as np

# found module but no type hints or library stubs
from scipy.special import binom  # type: ignore


def fdiff_coef(d: float, window: int) -> np.ndarray:
    """Return sequence of coefficients in fracdiff operator.

    Parameters
    ----------
    d : float
        Order of differentiation.
    window : int
        Number of terms.

    Returns
    -------
    coef : numpy.array, shape (window,)
        Coefficients in fracdiff operator.

    Examples
    --------
    >>> from sktime.libs.fracdiff import fdiff_coef
    >>> fdiff_coef(0.5, 4)
    array([ 1.    , -0.5   , -0.125 , -0.0625])
    >>> fdiff_coef(1.0, 4)
    array([ 1., -1.,  0., -0.])
    >>> fdiff_coef(1.5, 4)
    array([ 1.    , -1.5   ,  0.375 ,  0.0625])
    """
    return (-1) ** np.arange(window) * binom(d, np.arange(window))


def fdiff(
    a: np.ndarray,
    n: float = 1.0,
    axis: int = -1,
    prepend: Optional[np.ndarray] = None,
    append: Optional[np.ndarray] = None,
    window: int = 10,
    mode: str = "same",
) -> np.ndarray:
    r"""Calculate the `n`-th differentiation along the given axis.

    Extension of ``numpy.diff`` to fractional differentiation for
    fractional differences.

    If ``n`` is an integer, this function
    returns the same output as ``numpy.diff``,
    in indices ``n:`` along the given axis.
    The remaining indices are filled with values identical to ``a``.

    This is for consistency with the fractional differentiation,
    which returns an array of the same shape as the input array.

    Parameters
    ----------
    a : array_like, coercible to a numpy array.
        The input array.

    n : float, default=1.0
        The order of differentiation.

    axis : int, default=-1
        The axis along which differentiation is performed, default is the last axis.

    prepend : array_like, optional
        Values to prepend to ``a`` along axis prior to performing the differentiation.
        Scalar values are expanded to arrays with length 1 in the direction of axis and
        the shape of the input array in along all other axes.
        Otherwise the dimension and shape must match ``a`` except along axis.

    append : array_like, optional
        Values to append.

    window : int, default=10
        Number of observations to compute each element in the output.

    mode : {"same", "valid"}, default="same"

        "same" (default) :
            At the beginning of the time series,
            return elements where at least one coefficient of fracdiff is used.
            Output size along ``axis`` is :math:`L_{\\mathrm{in}}`
            where :math:`L_{\\mathrm{in}}` is the length of ``a`` along ``axis``
            (plus the lengths of ``append`` and ``prepend``).
            Boundary effects may be seen at the at the beginning of a time-series.

        "valid" :
            Return elements where all coefficients of fracdiff are used.
            Output size along ``axis`` is
            :math:`L_{\\mathrm{in}} - \\mathrm{window} + 1` where
            where :math:`L_{\\mathrm{in}}` is the length of ``a`` along ``axis``
            (plus the lengths of ``append`` and ``prepend``).
            Boundary effects are not seen.

    Returns
    -------
    fdiff : ``numpy.ndarray`` of same shape as ``a``
        The fractional differentiation.

    Examples
    --------
    This returns the same result with ``numpy.diff`` for integer `n`.

    >>> from sktime.libs.fracdiff import fdiff
    >>> a = np.array([1, 2, 4, 7, 0])
    >>> (np.diff(a) == fdiff(a)).all()
    True
    >>> (np.diff(a, 2) == fdiff(a, 2)).all()
    True

    This returns fractional differentiation for noninteger `n`.

    >>> fdiff(a, 0.5, window=3)
    array([ 1.   ,  1.5  ,  2.875,  4.75 , -4.   ])

    Mode "valid" returns elements for which all coefficients are convoluted.

    >>> fdiff(a, 0.5, window=3, mode="valid")
    array([ 2.875,  4.75 , -4.   ])
    >>> fdiff(a, 0.5, window=3, mode="valid", prepend=[1, 1])
    array([ 0.375,  1.375,  2.875,  4.75 , -4.   ])

    Differentiation along desired axis.

    >>> a = np.array([[  1,  3,  6, 10, 15],
    ...               [  0,  5,  6,  8, 11]])
    >>> fdiff(a, 0.5, window=3)
    array([[1.   , 2.5  , 4.375, 6.625, 9.25 ],
           [0.   , 5.   , 3.5  , 4.375, 6.25 ]])
    >>> fdiff(a, 0.5, window=3, axis=0)
    array([[ 1. ,  3. ,  6. , 10. , 15. ],
           [-0.5,  3.5,  3. ,  3. ,  3.5]])
    """
    a = np.asanyarray(a)
    # Mypy complains:
    # fracdiff/fdiff.py:135: error: Module has no attribute "normalize_axis_index"
    axis = np.core.multiarray.normalize_axis_index(axis, a.ndim)  # type: ignore
    dtype = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64

    a = _combine_pre_append(a, prepend, append, axis)

    if mode == "full":
        mode = "same"
        raise DeprecationWarning("mode 'full' was renamed to 'same'.")

    if a.ndim == 0:
        raise ValueError("diff requires input that is at least one dimensional")

    if mode == "valid":
        D = partial(np.convolve, fdiff_coef(n, window).astype(dtype), mode="valid")
        a = np.apply_along_axis(D, axis, a)
    elif mode == "same":
        # Convolve with the mode 'full' and cut last
        D = partial(np.convolve, fdiff_coef(n, window).astype(dtype), mode="full")
        s = tuple(
            slice(a.shape[axis]) if i == axis else slice(None) for i in range(a.ndim)
        )
        a = np.apply_along_axis(D, axis, a)
        a = a[s]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return a


def _combine_pre_append(a, prepend, append, axis):
    combined = []
    if prepend is not None:
        prepend = np.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not None:
        append = np.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = np.concatenate(combined, axis)

    return a
