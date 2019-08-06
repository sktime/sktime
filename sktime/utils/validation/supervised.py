import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length

__author__ = "Markus Löning"
__all__ = ["validate_X", "check_X_is_univariate", "validate_y", "validate_X_y"]


def validate_X(X):
    """Validate input data.

    Parameters
    ----------
    X : pandas DataFrame

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If X is an invalid input
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be a pandas.DataFrame, but found:"
                         f"{(type(X))}")


def validate_y(y):
    """Validate input data.

    Parameters
    ----------
    y : pandas Series or numpy ndarray

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If y is an invalid input
    """
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError(f"y must be either a pandas.Series or a numpy.ndarray, "
                         f"but found type: {type(y)}")


def validate_X_y(X, y):
    """Validate input data.

    Parameters
    ----------
    X : pandas DataFrame
    y : pandas Series or numpy ndarray

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If y is an invalid input
    """

    validate_X(X)
    validate_y(y)
    check_consistent_length(X, y)


def check_X_is_univariate(X):
    """Validate input data.

    Parameters
    ----------
    X : pandas DataFrame

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If X has more than one column
    """
    if X.shape[1] > 1:
        raise NotImplementedError(f"X must be univariate with X.shape[1] == 1, "
                                  f"but found: X.shape[1] == {X.shape[1]}. For "
                                  f"multivariate please use compositor classes. "
                                  f"Estimator-specific multivariate approaches are "
                                  f"not implemented yet.")
