__author__ = ["Markus Löning"]
__all__ = [
    "check_X",
    "check_y",
    "check_X_y",
    "_enforce_X_univariate",
    "_enforce_min_instances"
]

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length


def check_X(X, enforce_univariate=False, enforce_min_instances=1):
    """Validate input data.

    Parameters
    ----------
    X : pd.DataFrame
    enforce_univariate : bool, optional (default=False)
        Enforce that X is univariate.
    enforce_min_instances : int, optional (default=1)
        Enforce minimum number of instances.

    Returns
    -------
    X : pd.DataFrame

    Raises
    ------
    ValueError
        If X is an invalid input
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be a pd.DataFrame, but found: "
                         f"{(type(X))}")
    if enforce_univariate:
        _enforce_X_univariate(X)
    if enforce_min_instances > 0:
        _enforce_min_instances(X, min_instances=enforce_min_instances)
    return X


def check_y(y, enforce_min_instances=1):
    """Validate input data.

    Parameters
    ----------
    y : pd.Series or np.array
    enforce_min_instances : int, optional (default=1)
        Enforce minimum number of instances.


    Returns
    -------
    y : pd.Series or np.array

    Raises
    ------
    ValueError
        If y is an invalid input
    """
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError(
            f"y must be either a pd.Series or a np.ndarray, "
            f"but found type: {type(y)}")

    if enforce_min_instances > 0:
        _enforce_min_instances(y, min_instances=enforce_min_instances)

    return y


def check_X_y(X, y, enforce_univariate=False, enforce_min_instances=1):
    """Validate input data.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series or np.array
    enforce_univariate : bool, optional (default=False)
        Enforce that X is univariate.
    enforce_min_instances : int, optional (default=1)
        Enforce minimum number of instances.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series

    Raises
    ------
    ValueError
        If y is an invalid input
    """
    X = check_X(X, enforce_univariate=enforce_univariate)
    y = check_y(y)
    check_consistent_length(X, y)

    # Since we have already checked for consistent lengths, we only need to
    # check of the data containers for the minimum number of instances
    if enforce_min_instances > 0:
        _enforce_min_instances(y, min_instances=enforce_min_instances)

    return X, y


def _enforce_X_univariate(X):
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
        raise ValueError(
            f"X must be univariate with X.shape[1] == 1, "
            f"but found: X.shape[1] == {X.shape[1]}. For "
            f"multivariate problems please consider compositor classes. "
            f"Estimator-specific multivariate approaches are "
            f"not implemented yet.")


def _enforce_min_instances(x, min_instances=1):
    if hasattr(x, "shape"):
        n_instances = x.shape[0]
    else:
        x = np.asarray(x)
        n_instances = x.shape[0]

    if min_instances > 0:
        if n_instances < min_instances:
            raise ValueError(
                f"Found array with {n_instances} instance(s) "
                f"but a minimum of {min_instances} is required.")
