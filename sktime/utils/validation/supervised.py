import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length


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


def validate_univariate_X(X):
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
        raise ValueError(f"X must be univariate with X.shape[1] == 1, "
                         f"but found: X.shape[1] == {X.shape[1]}")
