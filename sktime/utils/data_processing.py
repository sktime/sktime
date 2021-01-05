# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

__all__ = [
    "from_3d_numpy_to_2d_array",
    "from_3d_numpy_to_nested",
    "from_nested_to_2d_array",
    "from_2d_array_to_nested",
    "from_nested_to_3d_numpy",
    "from_nested_to_long",
    "is_nested_dataframe",
]


def _check_equal_index(X):
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
    # TODO assumes columns are typed (i.e. all rows for a given column have
    #  the same type)
    # TODO only handles series columns, raises error for columns with
    #  primitives

    indexes = []
    # Check index for each column separately.
    for c, col in enumerate(X.columns):

        # Get index from first row, can be either pd.Series or np.array.
        first_index = (
            X.iloc[0, c].index
            if hasattr(X.iloc[0, c], "index")
            else np.arange(X.iloc[c, 0].shape[0])
        )

        # Series must contain at least 2 observations, otherwise should be
        # primitive.
        if len(first_index) < 2:
            raise ValueError(
                f"Time series must contain at least 2 observations, "
                f"but found: "
                f"{len(first_index)} observations in column: {col}"
            )

        # Check index for all rows.
        for i in range(1, X.shape[0]):
            index = (
                X.iloc[i, c].index
                if hasattr(X.iloc[i, c], "index")
                else np.arange(X.iloc[c, 0].shape[0])
            )
            if not np.array_equal(first_index, index):
                raise ValueError(
                    f"Found time series with unequal index in column {col}. "
                    f"Input time-series must have the same index."
                )
        indexes.append(first_index)

    return indexes


def from_3d_numpy_to_2d_array(X):
    return X.reshape(X.shape[0], -1)


def from_nested_to_2d_array(X, return_numpy=False):
    """Convert nested pandas DataFrames or Series with numpy arrays or
    pandas Series in cells into tabular
    pandas DataFrame with primitives in cells, i.e. a data frame with the
    same number of rows as the input data and
    as many columns as there are observations in the nested series. Requires
    series to be have the same index.

    Parameters
    ----------
    X : nested pandas DataFrame or nested Series
    return_numpy : bool, optional (default=False)
        - If True, returns a numpy array of the tabular data.
        - If False, returns a pandas dataframe with row and column names.

    Returns
    -------
     Xt : pandas DataFrame
        Transformed dataframe in tabular format
    """

    # TODO does not handle dataframes with nested series columns *and*
    #  standard columns containing only primitives

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

        if Xt.ndim != 2:
            raise ValueError(
                "Tabularization failed, it's possible that not "
                "all series were of equal length"
            )

    else:
        raise ValueError(
            f"Expected input is pandas Series or pandas DataFrame, "
            f"but found: {type(X)}"
        )

    if return_numpy:
        return Xt

    Xt = pd.DataFrame(Xt)

    # create column names from time index
    if X.ndim == 1:
        time_index = (
            X.iloc[0].index
            if hasattr(X.iloc[0], "index")
            else np.arange(X.iloc[0].shape[0])
        )
        columns = [f"{X.name}__{i}" for i in time_index]

    else:
        columns = []
        for colname, col in X.items():
            time_index = (
                col.iloc[0].index
                if hasattr(col.iloc[0], "index")
                else np.arange(col.iloc[0].shape[0])
            )
            columns.extend([f"{colname}__{i}" for i in time_index])

    Xt.index = X.index
    Xt.columns = columns
    return Xt


def from_2d_array_to_nested(
    X, index=None, columns=None, time_index=None, cells_as_numpy=False
):
    """Convert tabular pandas DataFrame with only primitives in cells into
    nested pandas DataFrame with a single column.

    Parameters
    ----------
    X : pandas DataFrame
    cells_as_numpy : bool, optional (default=False)
        - If True, returns a numpy arrays within cells of nested pandas
        DataFrame.
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

    if (time_index is not None) and cells_as_numpy:
        raise ValueError(
            "`Time_index` cannot be specified when `return_arrays` is True, "
            "time index can only be set to "
            "pandas Series"
        )
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    container = np.array if cells_as_numpy else pd.Series

    # for 2d numpy array, rows represent instances, columns represent time points
    n_instances, n_timepoints = X.shape

    if time_index is None:
        time_index = np.arange(n_timepoints)
    kwargs = {"index": time_index}

    Xt = pd.DataFrame(
        pd.Series([container(X[i, :], **kwargs) for i in range(n_instances)])
    )
    if index is not None:
        Xt.index = index
    if columns is not None:
        Xt.columns = columns
    return Xt


def _concat_nested_arrays(arrs, return_arrays=False):
    """
    Helper function to nest tabular arrays from nested list of arrays.

    Parameters
    ----------
    arrs : list of numpy arrays
        Arrays must have the same number of rows, but can have varying
        number of columns.
    return_arrays: bool, optional (default=False)
        - If True, return pandas DataFrame with nested numpy arrays.
        - If False, return pandas DataFrame with nested pandas Series.

    Returns
    -------
    Xt : pandas DataFrame
        Transformed dataframe with nested column for each input array.
    """
    if return_arrays:
        Xt = pd.DataFrame(
            np.column_stack(
                [pd.Series([np.array(vals) for vals in interval]) for interval in arrs]
            )
        )
    else:
        Xt = pd.DataFrame(
            np.column_stack(
                [pd.Series([pd.Series(vals) for vals in interval]) for interval in arrs]
            )
        )
    return Xt


def _get_time_index(X):
    """Helper function to get index of time series data

    Parameters
    ----------
    X : pd.DataFrame

    Returns
    -------
    time_index : pandas Index
        Index of time series
    """

    def _get_index(x):
        if hasattr(x, "index"):
            return x.index
        else:
            # select last dimension for time index
            return pd.RangeIndex(x.shape[-1])

    # assumes that all samples share the same the time index, only looks at
    # first row
    if isinstance(X, pd.DataFrame):
        return _get_index(X.iloc[0, 0])

    elif isinstance(X, pd.Series):
        return _get_index(X.iloc[0])

    elif isinstance(X, np.ndarray):
        return _get_index(X)

    else:
        raise ValueError(
            f"X must be a pandas DataFrame or Series, but found: {type(X)}"
        )


def _get_column_names(X):
    if isinstance(X, pd.DataFrame):
        return X.columns
    else:
        return [f"col{i}" for i in range(X.shape[1])]


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
        df = from_nested_to_2d_array(X.iloc[:, i])
        df = df.reset_index()
        df = df.melt(id_vars="index")
        df["column"] = df["variable"].str.split("__").str[0]
        df["time_index"] = df["variable"].str.split("__").str[1]
        df = df.drop(columns="variable")
        columns.append(df)
    return pd.concat(columns)


def from_nested_to_3d_numpy(X):
    """Convert pandas DataFrame (with time series as pandas Series in cells)
    into NumPy ndarray with shape (n_instances, n_columns, n_timepoints).

    Parameters
    ----------
    X : pd.DataFrame
        Nested pandas DataFrame

    Returns
    -------
    X : np.ndarrray
        3-dimensional NumPy array
    """
    # n_instances, n_columns = X.shape
    # n_timepoints = X.iloc[0, 0].shape[0]
    # array = np.empty((n_instances, n_columns, n_timepoints))
    # for column in range(n_columns):
    #     array[:, column, :] = X.iloc[:, column].tolist()
    # return array
    return np.stack(
        X.applymap(lambda cell: cell.to_numpy())
        .apply(lambda row: np.stack(row), axis=1)
        .to_numpy()
    )


def from_3d_numpy_to_nested(X):
    """Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints)
    into pandas DataFrame (with time series as pandas Series in cells)

    Parameters
    ----------
    X : NumPy ndarray, input

    Returns
    -------
    pandas DataFrame
    """
    df = pd.DataFrame()
    n_instances = X.shape[0]
    n_variables = X.shape[1]
    for variable in range(n_variables):
        df["var_" + str(variable)] = [
            pd.Series(X[instance][variable]) for instance in range(n_instances)
        ]
    return df


def is_nested_dataframe(X):
    return isinstance(X, pd.DataFrame) and isinstance(
        X.iloc[0, 0], (np.ndarray, pd.Series)
    )
