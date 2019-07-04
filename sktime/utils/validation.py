import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

__all__ = ['check_is_fitted_in_transform',
           'check_ts_array',
           'check_equal_index',
           'check_consistent_indices',
           'check_X_y',
           'check_ts_X_y',
           'validate_time_index',
           'validate_sp',
           'validate_fh']


def check_X_y(instances, class_labels=None):
    if not isinstance(instances, pd.DataFrame):
        raise ValueError("instances not in panda dataframe")
    if class_labels is not None:
        if len(class_labels) != instances.shape[0]:
            raise ValueError("instances not same length as class_labels")


def check_ts_X_y(X, y):
    """
    Placeholder function for input validation.
    """
    # TODO: add proper checks (e.g. check if input stuff is pandas full of objects)
    # currently it checks neither the data nor the datatype
    # return check_X_y(X, y, dtype=None, ensure_2d=False)
    return X, y


def check_ts_array(X):
    """Helper function for input checks"""
    # if not isinstance(X, pd.DataFrame):
    #     raise ValueError(f"Input data must be pandas DataFrame, but found: {type(X)}")
    #
    # if X.shape[1] > 1:
    #     raise NotImplementedError(f"Input data must be a univariate pandas DataFrame with a single column, "
    #                               f"but found: {X.shape[1]} columns")
    return X


def check_equal_index(X):
    """
    Check if all time-series for a given column in a
    nested pandas DataFrame have the same index.

    Parameters
    ----------
    X : nested pandas DataFrame
        Input dataframe with time-series in cells.

    Returns
    -------
    indexes : list of indixes
        List of indixes with one index for each column
    """
    # TODO handle 1d series, not only 2d dataframes
    # TODO assumes columns are typed (i.e. all rows for a given column have the same type)
    # TODO only handles series columns, raises error for columns with primitives

    indexes = []
    # Check index for each column separately.
    for c, col in enumerate(X.columns):

        # Get index from first row, can be either pd.Series or np.array.
        first_index = X.iloc[0, c].index if hasattr(X.iloc[0, c], 'index') else np.arange(X.iloc[c, 0].shape[0])

        # Series must contain at least 2 observations, otherwise should be primitive.
        if len(first_index) < 2:
            raise ValueError(f'Time series must contain at least 2 observations, but found: '
                             f'{len(first_index)} observations in column: {col}')

        # Check index for all rows.
        for i in range(1, X.shape[0]):
            index = X.iloc[i, c].index if hasattr(X.iloc[i, c], 'index') else np.arange(X.iloc[c, 0].shape[0])
            if not np.array_equal(first_index, index):
                raise ValueError(f'Found time series with unequal index in column {col}. '
                                 f'Input time-series must have the same index.')
        indexes.append(first_index)

    return indexes


def validate_sp(sp):
    """Validate seasonal periodicity.

    Parameters
    ----------
    sp : int
        Seasonal periodicity

    Returns
    -------
    sp : int
        Validated seasonal periodicity
    """

    if sp is None:
        return sp

    else:
        if not isinstance(sp, int) and (sp >= 0):
            raise ValueError(f"Seasonal periodicity (sp) has to be a positive integer, but found: "
                             f"{sp} of type: {type(sp)}")
        return sp


def validate_fh(fh):
    """Validate forecasting horizon.

    Parameters
    ----------
    fh : int or list of int
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


def check_is_fitted_in_transform(estimator, attributes, msg=None, all_or_any=all):
    """Checks if the estimator is fitted during transform by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``
    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    Returns
    -------
    None
    
    Raises
    ------
    NotFittedError
        If the attributes are not found.    
    """
    if msg is None:
        msg = ("This %(name)s instance has not been fitted yet. Call 'transform' with "
               "appropriate arguments before using this method.")

    check_is_fitted(estimator, attributes=attributes, msg=msg, all_or_any=all_or_any)


def validate_time_index(time_index):
    """Validate time index

    Parameters
    ----------
    time_index : array-like

    Returns
    -------
    time_index : ndarray
    """
    # period or datetime index are not support yet
    # TODO add support for period/datetime indexing
    if isinstance(time_index, (pd.PeriodIndex, pd.DatetimeIndex)):
        raise NotImplementedError(f"{type(time_index)} is not fully supported yet, "
                                  f"use pandas RangeIndex instead")

    return np.asarray(time_index)


def check_consistent_indices(x, y):
    """Check that x and y have consistent indices.

    Parameters
    ----------
    x : pandas Series
    y : pandas Series

    Raises:
    -------
    ValueError
        If indicies are not equal
    """
    if not x.index.equals(y.index):
        raise ValueError(f"Found input variables with inconsistent indices")
