"""Validation functions for input data series."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def check_data(
    X: pd.DataFrame | pd.Series | ArrayLike,
    min_length: int,
    min_length_name: str = "min_length",
    allow_missing_values: bool = False,
) -> pd.DataFrame:
    """Check if input data is valid.

    Parameters
    ----------
    X : pd.DataFrame, pd.Series, np.ndarray
        Input data to check.
    min_length : int
        Minimum number of samples in X.
    min_length_name : str, optional (default="min_length")
        Name of min_length parameter to be shown in the error message.
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


def as_2d_array(X: ArrayLike, vector_as_column=True, dtype=None) -> np.ndarray:
    """Convert an array-like object to a 2D numpy array.

    Parameters
    ----------
    X : `ArrayLike`
        Array-like object.

    Returns
    -------
    X : `np.ndarray`
        2D numpy array.
    """
    X = np.asarray(X, dtype=dtype)
    if X.ndim == 1:
        X = X.reshape(-1, 1) if vector_as_column else X.reshape(1, -1)
    elif X.ndim > 2:
        raise ValueError("X must be at most 2-dimensional.")
    return X
