import numpy as np
import pandas as pd

from sktime.utils.validation import check_ts_array


def select_times(X, times):
    """Select times from time series within cells of nested pandas DataFrame.

    Parameters
    ----------
    X : nested pandas DataFrame or nested Series
    times : numpy ndarray of times to select from time series

    Returns
    -------
    Xt : pandas DataFrame
        pandas DataFrame in nested format containing only selected times
    """
    # TODO currently we loose the time index, need to add it back to Xt after slicing in time
    Xt = detabularise(tabularise(X).iloc[:, times])
    Xt.columns = X.columns
    return Xt


def tabularize(X, return_array=False):
    """Convert nested pandas DataFrames or Series with numpy arrays or pandas Series in cells into tabular
    pandas DataFrame with primitives in cells, i.e. a data frame with the same number of rows as the input data and
    as many columns as there are observations in the nested series. Requires series to be have the same index.

    Parameters
    ----------
    X : nested pandas DataFrame or nested Series
    return_array : bool, optional (default=False)
        - If True, returns a numpy array of the tabular data.
        - If False, returns a pandas dataframe with row and column names.

    Returns
    -------
     Xt : pandas DataFrame
        Transformed dataframe in tabular format
    """

    # TODO does not handle dataframes with nested series columns *and* standard columns containing only primitives

    # convert nested data into tabular data
    if X.ndim == 1:
        Xt = np.array(X.tolist())
    else:
        Xt = np.hstack([col.tolist() for _, col in X.items()])

    if return_array:
        return Xt

    Xt = pd.DataFrame(Xt)

    # create column names from time index
    if X.ndim == 1:
        time_index = X.iloc[0].index if hasattr(X.iloc[0], 'index') else np.arange(X.iloc[0].shape[0])
        columns = [f'{X.name}__{i}' for i in time_index]

    else:
        columns = []
        for colname, col in X.items():
            time_index = col.iloc[0].index if hasattr(col.iloc[0], 'index') else np.arange(col.iloc[0].shape[0])
            columns.extend([f'{colname}__{i}' for i in time_index])

    Xt.index = X.index
    Xt.columns = columns
    return Xt


def detabularize(X, index=None, time_index=None, return_arrays=False):
    """Convert tabular pandas DataFrame with only primitives in cells into nested pandas DataFrame with a single column.

    Parameters
    ----------
    X : pandas DataFrame
    return_arrays : bool, optional (default=False)
        - If True, returns a numpy arrays within cells of nested pandas DataFrame.
        - If False, returns a pandas Series within cells.
    index : array-like, shape=[n_samples], optional (default=None)
        Sample (row) index of transformed dataframe
    time_index : array-like, shape=[n_obs], optional (default=None)
        Time series index of transformed dataframe

    Returns
    -------
    Xt : pandas DataFrame
        Transformed dataframe in nested format
    """

    if (time_index is not None) and return_arrays:
        raise ValueError("`Time_index` cannot be specified when `return_arrays` is True, time index can only be set to "
                         "pandas Series")

    container = np.array if return_arrays else pd.Series

    n_samples, n_obs = X.shape

    if time_index is None:
        time_index = np.arange(n_obs)
    kwargs = {'index': time_index}

    Xt = pd.DataFrame(pd.Series([container(X.iloc[i, :].values, **kwargs) for i in range(n_samples)]))

    if index is not None:
         Xt.index = index

    return Xt


tabularise = tabularize


detabularise = detabularize


def concat_nested_arrays(arrs, return_arrays=False):
    """
    Helper function to nest tabular arrays from nested list of arrays.

    Parameters
    ----------
    arrs : list of numpy arrays
        Arrays must have the same number of rows, but can have varying number of columns.
    return_arrays: bool, optional (default=False)
        - If True, return pandas DataFrame with nested numpy arrays.
        - If False, return pandas DataFrame with nested pandas Series.

    Returns
    -------
    Xt : pandas DataFrame
        Transformed dataframe with nested column for each input array.
    """
    if return_arrays:
        Xt = pd.DataFrame(np.column_stack(
            [pd.Series([np.array(vals) for vals in interval])
             for interval in arrs]))
    else:
        Xt = pd.DataFrame(np.column_stack(
            [pd.Series([pd.Series(vals) for vals in interval])
             for interval in arrs]))
    return Xt


def get_time_index(X):
    """Helper function to get index of time series data

    Parameters
    ----------
    X : pandas DataFrame

    Returns
    -------
    time_index : pandas Index
        Index of time series
    """

    # assumes that all samples share the same the time index
    if isinstance(X, pd.DataFrame):
        X = check_ts_array(X)
        Xs = X.iloc[0, 0]

    elif isinstance(X, pd.Series):
        Xs = X.iloc[0]

    else:
        raise ValueError(f"X must be a pandas DataFrame or Series, but found: {type(X)}")

    # get time index
    time_index = Xs.index if hasattr(Xs, 'index') else pd.RangeIndex(Xs.shape[0])

    # check time index
    if isinstance(time_index, (pd.PeriodIndex, pd.DatetimeIndex)):
        raise NotImplementedError(f"{type(time_index)} is not supported yet, "
                                  f"use pandas RangeIndex instead")

    return time_index