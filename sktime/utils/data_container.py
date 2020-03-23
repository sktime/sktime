import numpy as np
import pandas as pd


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
    if isinstance(X, pd.Series):
        Xt = np.array(X.tolist())

    elif isinstance(X, pd.DataFrame):
        try:
            Xt = np.hstack([X.iloc[:, i].tolist() for i in range(X.shape[1])])

        # except strange key error for specific case
        except KeyError:
            if (X.shape == (1, 1)) and (X.iloc[0, 0].shape == (1,)):
                # in fact only breaks when an additional condition is met,
                # namely that the index of the time series of a single value
                # does not start with 0, e.g. pd.RangeIndex(9, 10) as is the
                # case in forecasting
                Xt = X.iloc[0, 0].values
            else:
                raise

    else:
        raise ValueError(f"Expected input is pandas Series or pandas DataFrame, "
                         f"but found: {type(X)}")

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

    # assumes that all samples share the same the time index, only looks at first row
    if isinstance(X, pd.DataFrame):
        Xs = X.iloc[0, 0]

    elif isinstance(X, pd.Series):
        Xs = X.iloc[0]

    else:
        raise ValueError(f"X must be a pandas DataFrame or Series, but found: {type(X)}")

    # get time index
    time_index = Xs.index if hasattr(Xs, 'index') else pd.RangeIndex(Xs.shape[0])

    return time_index


def from_nested_to_long(X):
    """Convert nested dataframe to long dataframe

    Parameters
    ----------
    X : pd.DataFrame
        nested dataframe

    Returns
    -------
    Xt : pd.DataFrame
        long dataframe
    """
    columns = []

    for i in range(len(X.columns)):
        df = tabularise(X.iloc[:, i])
        df = df.reset_index()
        df = df.melt(id_vars="index")
        df["column"] = df["variable"].str.split("__").str[0]
        df["time_index"] = df["variable"].str.split("__").str[1]
        df = df.drop(columns="variable")
        columns.append(df)
    return pd.concat(columns)


def nested_to_3d_numpy(X, a=None, b=None):
    """Convert pandas DataFrame (with time series as pandas Series in cells) into NumPy ndarray with shape (n_instances, n_columns, n_timepoints).

    Parameters
    ----------
    X : pandas DataFrame, input
    a : int, first row (optional, default None)
    b : int, last row (optional, default None)

    Returns
    -------
    NumPy ndarray, converted NumPy ndarray
    """
    return np.stack(
        X.iloc[a:b].applymap(lambda cell: cell.to_numpy()).apply(lambda row: np.stack(row), axis=1).to_numpy())
