# -*- coding: utf-8 -*-
"""
Functions to process data from standard numpy formats into

TO DO
1. Sort out the maintenance import
2. resolve naming: change "nested" to sktime_format?
"""
__all__ = ["from_3d_numpy_to_2d_array",
           "from_nested_to_2d_array",
           "from_2d_array_to_nested",
           "from_nested_to_long",
           "from_nested_to_3d_numpy",
           "from_3d_numpy_to_nested",
           "from_long_to_nested",
           "generate_example_long_table"
           ]
__author__ = ["Jason Lines", "Tony Bagnall"]


import numpy as np
import pandas as pd

from sktime.utils._maintenance import deprecated


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


@deprecated("Please use `from_nested_to_2d_array` instead.")
def tabularize(X, return_array=False):
    return from_nested_to_2d_array(X, return_array)


@deprecated("Please use `from_2d_array_to_nested` instead.")
def detabularize(X, return_array=False):
    return from_2d_array_to_nested(X, return_array)


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


# assumes data is in a long table format with the following structure:
#      | case_id | dim_id | reading_id | value
# ------------------------------------------------
#   0  |   int   |  int   |    int     | double
#   1  |   int   |  int   |    int     | double
#   2  |   int   |  int   |    int     | double
#   3  |   int   |  int   |    int     | double
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


def from_long_to_nested(long_dataframe):
    # get distinct dimension ids
    unique_dim_ids = long_dataframe.iloc[:, 1].unique()
    num_dims = len(unique_dim_ids)

    data_by_dim = []
    indices = []

    # get number of distinct cases (note: a case may have 1 or many dimensions)
    unique_case_ids = long_dataframe.iloc[:, 0].unique()
    # assume series are indexed from 0 to m-1 (can map to non-linear indices
    # later if needed)

    # init a list of size m for each d - to store the series data for m
    # cases over d dimensions
    # also, data may not be in order in long format so store index data for
    # aligning output later
    # (i.e. two stores required: one for reading id/timestamp and one for
    # value)
    for d in range(0, num_dims):
        data_by_dim.append([])
        indices.append([])
        for _c in range(0, len(unique_case_ids)):
            data_by_dim[d].append([])
            indices[d].append([])

    # go through every row in the dataframe
    for i in range(0, len(long_dataframe)):
        # extract the relevant data, catch cases where the dim id is not an
        # int as it must be the class

        row = long_dataframe.iloc[i]
        case_id = int(row[0])
        dim_id = int(row[1])
        reading_id = int(row[2])
        value = row[3]
        data_by_dim[dim_id][case_id].append(value)
        indices[dim_id][case_id].append(reading_id)

    x_data = {}
    for d in range(0, num_dims):
        key = "dim_" + str(d)
        dim_list = []
        for i in range(0, len(unique_case_ids)):
            temp = pd.Series(data_by_dim[d][i], indices[d][i])
            dim_list.append(temp)
        x_data[key] = pd.Series(dim_list)

    return pd.DataFrame(x_data)


def generate_example_long_table(num_cases=50, series_len=20, num_dims=2):
    rows_per_case = series_len * num_dims
    total_rows = num_cases * series_len * num_dims

    case_ids = np.empty(total_rows, dtype=np.int)
    idxs = np.empty(total_rows, dtype=np.int)
    dims = np.empty(total_rows, dtype=np.int)
    vals = np.random.rand(total_rows)

    for i in range(total_rows):
        case_ids[i] = int(i / rows_per_case)
        rem = i % rows_per_case
        dims[i] = int(rem / series_len)
        idxs[i] = rem % series_len

    df = pd.DataFrame()
    df["case_id"] = pd.Series(case_ids)
    df["dim_id"] = pd.Series(dims)
    df["reading_id"] = pd.Series(idxs)
    df["value"] = pd.Series(vals)
    return df