"""Utility functions for cost calculations."""

import numbers

import numpy as np
from numpy.typing import ArrayLike

MeanType = ArrayLike | numbers.Number
VarType = ArrayLike | numbers.Number
CovType = ArrayLike | numbers.Number


def check_mean(mean: MeanType, X: np.ndarray) -> np.ndarray:
    """Check if the fixed mean parameter is valid.

    Parameters
    ----------
    mean : np.ndarray or numbers.Number
        Fixed mean for the cost calculation.
    X : np.ndarray
        2d input data.

    Returns
    -------
    mean : np.ndarray
        Fixed mean for the cost calculation.
    """
    mean = np.array([mean]) if isinstance(mean, numbers.Number) else np.asarray(mean)
    if len(mean) != 1 and len(mean) != X.shape[1]:
        raise ValueError(f"mean must have length 1 or X.shape[1], got {len(mean)}.")
    return mean


def check_non_negative_parameter(scale: MeanType, X: np.ndarray) -> np.ndarray:
    """Check if the fixed scale parameter is valid.

    Parameters
    ----------
    scale(s) : np.ndarray or numbers.Number
        Fixed scale(s) for the cost calculation.
    X : np.ndarray
        2d input data.

    Returns
    -------
    scale(s) : np.ndarray
        Fixed scales for cost calculation.
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


def check_var(var: VarType, X: np.ndarray) -> np.ndarray:
    """Check if the fixed variance parameter is valid.

    Parameters
    ----------
    var : np.ndarray or numbers.Number
        Fixed variance for the cost calculation.
    X : np.ndarray
        2d input data.

    Returns
    -------
    var : np.ndarray
        Fixed variance for the cost calculation.
    """
    var = np.array([var]) if isinstance(var, numbers.Number) else np.asarray(var)
    if len(var) != 1 and len(var) != X.shape[1]:
        raise ValueError(f"var must have length 1 or X.shape[1], got {len(var)}.")

    if np.any(var <= 0):
        raise ValueError("var must be positive.")
    return var


def check_cov(cov: CovType, X: np.ndarray, force_float: bool = False) -> np.ndarray:
    """Check if the fixed covariance matrix parameter is valid.

    Parameters
    ----------
    cov : np.ndarray
        Fixed covariance matrix for the cost calculation.
    X : np.ndarray
        2d input data.
    force_float : bool, default=False
        If True, force the covariance matrix to be of
        floating point data type.

    Returns
    -------
    cov : np.ndarray
        Fixed covariance matrix for the cost calculation.
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
