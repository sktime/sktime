import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length


def check_X(X):
    """Validate input data.

    Parameters
    ----------
    X : pandas DataFrame
        input data

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


def check_y(y):
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


def check_X_y(X, y):
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

    check_X(X)
    check_y(y)
    check_consistent_length(X, y)


def check_univariate_X(X):
    if X.shape[1] > 1:
        raise ValueError(f"X must be univariate with X.shape[1] == 1, "
                         f"but found: X.shape[1] == {X.shape[1]}")


def validate_fh(fh):
    """
    Validate forecasting horizon.

    Parameters
    ----------
    fh : list of int
        Forecasting horizon with steps ahead to predict.

    Returns
    -------
    fh : numpy array of int
        Sorted and validated forecasting horizon.
    """

    # Set default as one-step ahead
    if fh is None:
        return np.ones(1, dtype=np.int)

    # Check single integer
    elif np.issubdtype(type(fh), np.integer):
        return np.array([fh], dtype=np.int)

    # Check array-like input
    else:
        if isinstance(fh, list):
            if len(fh) < 1:
                raise ValueError(f"`fh` must specify at least one step, but found: "
                                 f"{type(fh)} of length {len(fh)}")
            if not np.all([np.issubdtype(type(h), np.integer) for h in fh]):
                raise ValueError('If `fh` is passed as a list, '
                                 'it has to be a list of integers')

        elif isinstance(fh, np.ndarray):
            if fh.ndim > 1:
                raise ValueError(f"`fh` must be a 1d array, but found: "
                                 f"{fh.ndim} dimensions")
            if len(fh) < 1:
                raise ValueError(f"`fh` must specify at least one step, but found: "
                                 f"{type(fh)} of length {len(fh)}")
            if not np.issubdtype(fh.dtype, np.integer):
                raise ValueError(
                    f'If `fh` is passed as an array, it has to be an array of '
                    f'integers, but found an array of dtype: {fh.dtype}')

        else:
            raise ValueError(f"`fh` has to be either a list or array of integers, or a single "
                             f"integer, but found: {type(fh)}")

        return np.asarray(np.sort(fh), dtype=np.int)
