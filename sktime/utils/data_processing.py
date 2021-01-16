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
    "from_long_to_nested",
    "from_multi_index_to_3d_numpy",
    "from_3d_numpy_to_multi_index",
    "from_multi_index_to_nested",
    "from_nested_to_multi_index",
    "are_columns_nested",
    "is_nested_dataframe",
    "convert_from_dictionary",
]


def _cell_is_series_or_array(cell):
    return isinstance(cell, (pd.Series, np.ndarray))


def _nested_cell_mask(X):
    return X.applymap(_cell_is_series_or_array)


def are_columns_nested(X):
    """Checks whether any cells have nested structure in each DataFrame column.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to check for nested data structures.

    Returns
    -------
    any_nested : bool
        If True, at least one column is nested.
        If False, no nested columns.
    """
    any_nested = _nested_cell_mask(X).any().values
    return any_nested


def _nested_cell_timepoints(cell):
    if _cell_is_series_or_array(cell):
        n_timepoints = cell.shape[0]
    else:
        n_timepoints = 0
    return n_timepoints


def _check_equal_index(X):
    """
    Check if all time-series for a given column in a
    nested pandas DataFrame have the same index.

    Parameters
    ----------
    X : nested pd.DataFrame
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
    """Converts 3d NumPy array (n_instances, n_columns, n_timepoints) to
    a 2d NumPy array with shape (n_instances, n_columns*n_timepoints)

    Parameters
    ----------
    X : np.ndarray
        The input 3d-NumPy array with shape
        (n_instances, n_columns, n_timepoints)

    Returns
    -------
    array_2d : np.ndarray
        A 2d-NumPy array with shape (n_instances, n_columns*n_timepoints)
    """
    array_2d = X.reshape(X.shape[0], -1)
    return array_2d


def from_nested_to_2d_array(X, return_numpy=False):
    """Convert nested pandas DataFrame or Series with NumPy arrays or
    pandas Series in cells into tabular
    pandas DataFrame with primitives in cells, i.e. a data frame with the
    same number of rows as the input data and
    as many columns as there are observations in the nested series. Requires
    series to be have the same index.

    Parameters
    ----------
    X : nested pd.DataFrame or nested pd.Series
    return_numpy : bool, default = False
        - If True, returns a NumPy array of the tabular data.
        - If False, returns a pandas DataFrame with row and column names.

    Returns
    -------
     Xt : pandas DataFrame
        Transformed DataFrame in tabular format
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
    X : pd.DataFrame

    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series

    index : array-like, shape=[n_samples], optional (default = None)
        Sample (row) index of transformed DataFrame

    time_index : array-like, shape=[n_obs], optional (default = None)
        Time series index of transformed DataFrame

    Returns
    -------
    Xt : pd.DataFrame
        Transformed DataFrame in nested format
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


def convert_from_dictionary(ts_dict):
    """
    Simple conversion from a dictionary of each series, e.g. univariate
        x = {
            "Series1": [1.0,2.0,3.0,1.0,2.0],
            "Series2": [3.0,2.0,1.0,3.0,2.0],
        }
    or multivariate, e.g.
    to sktime panda format
    TODO: Adapt for multivariate
    """
    panda = pd.DataFrame(ts_dict)
    panda = panda.transpose()
    return from_2d_array_to_nested(panda)


def _concat_nested_arrays(arrs, cells_as_numpy=False):
    """
    Helper function to nest tabular arrays from nested list of arrays.

    Parameters
    ----------
    arrs : list of numpy arrays
        Arrays must have the same number of rows, but can have varying
        number of columns.

    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series

    Returns
    -------
    Xt : pandas DataFrame
        Transformed dataframe with nested column for each input array.
    """
    if cells_as_numpy:
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


def _get_index(x):
    if hasattr(x, "index"):
        return x.index
    else:
        # select last dimension for time index
        return pd.RangeIndex(x.shape[-1])


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


def _make_column_names(column_count):
    return [f"var_{i}" for i in range(column_count)]


def _get_column_names(X):
    if isinstance(X, pd.DataFrame):
        return X.columns
    else:
        return _make_column_names(X.shape[1])


def from_nested_to_long(
    X, instance_column_name=None, time_column_name=None, dimension_column_name=None
):
    """Convert nested DataFrame to long DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        The nested DataFrame

    instance_column_name : str
        The name of column corresponding to the DataFrame's instances

    time_column_name : str
        The name of the column corresponding to the DataFrame's timepoints.

    dimension_column_name : str
        The name of the column corresponding to the DataFrame's dimensions.

    Returns
    -------
    long_df : pd.DataFrame
        Long pandas DataFrame
    """

    long_df = from_nested_to_multi_index(
        X, instance_index="index", time_index="time_index"
    )
    long_df.reset_index(inplace=True)
    long_df = long_df.melt(id_vars=["index", "time_index"], var_name="column")

    col_rename_dict = {}
    if instance_column_name is not None:
        col_rename_dict["index"] = instance_column_name

    if time_column_name is not None:
        col_rename_dict["time_index"] = time_column_name

    if dimension_column_name is not None:
        col_rename_dict["column"] = dimension_column_name

    if len(col_rename_dict) > 0:
        long_df = long_df.rename(columns=col_rename_dict)

    return long_df


def from_long_to_nested(
    X_long,
    instance_column_name="case_id",
    time_column_name="reading_id",
    dimension_column_name="dim_id",
    value_column_name="value",
    column_names=None,
):
    """Convert long DataFrame to a nested DataFrame.

    Parameters
    ----------
    X_long : pd.DataFrame
        The long DataFrame

    instance_column_name : str, default = 'case_id'
        The name of column corresponding to the DataFrame's instances.

    time_column_name : str, default = 'reading_id'
        The name of the column corresponding to the DataFrame's timepoints.

    dimension_column_name : str, default = 'dim_id'
        The name of the column corresponding to the DataFrame's dimensions.

    value_column_name : str, default = 'value'
        The name of the column corresponding to the DataFrame's values.

    column_names : list, optional
        Optional list of column names to use for nested DataFrame columns.

    Returns
    -------
    X_nested : pd.DataFrame
        Nested pandas DataFrame
    """

    X_nested = X_long.pivot(
        index=[instance_column_name, time_column_name],
        columns=dimension_column_name,
        values=value_column_name,
    )
    X_nested = from_multi_index_to_nested(X_nested, instance_index=instance_column_name)

    n_columns = X_nested.shape[1]
    if column_names is None:
        X_nested.columns = _make_column_names(n_columns)

    else:
        X_nested.columns = column_names

    # # get distinct dimension ids
    # unique_dim_ids = long_dataframe.iloc[:, 1].unique()
    # num_dims = len(unique_dim_ids)

    # data_by_dim = []
    # indices = []

    # # get number of distinct cases (note: a case may have 1 or many dimensions)
    # unique_case_ids = long_dataframe.iloc[:, 0].unique()
    # # assume series are indexed from 0 to m-1 (can map to non-linear indices
    # # later if needed)

    # # init a list of size m for each d - to store the series data for m
    # # cases over d dimensions
    # # also, data may not be in order in long format so store index data for
    # # aligning output later
    # # (i.e. two stores required: one for reading id/timestamp and one for
    # # value)
    # for d in range(0, num_dims):
    #     data_by_dim.append([])
    #     indices.append([])
    #     for _c in range(0, len(unique_case_ids)):
    #         data_by_dim[d].append([])
    #         indices[d].append([])

    # # go through every row in the dataframe
    # for i in range(0, len(long_dataframe)):
    #     # extract the relevant data, catch cases where the dim id is not an
    #     # int as it must be the class

    #     row = long_dataframe.iloc[i]
    #     case_id = int(row[0])
    #     dim_id = int(row[1])
    #     reading_id = int(row[2])
    #     value = row[3]
    #     data_by_dim[dim_id][case_id].append(value)
    #     indices[dim_id][case_id].append(reading_id)

    # x_data = {}
    # for d in range(0, num_dims):
    #     key = "dim_" + str(d)
    #     dim_list = []
    #     for i in range(0, len(unique_case_ids)):
    #         temp = pd.Series(data_by_dim[d][i], indices[d][i])
    #         dim_list.append(temp)
    #     x_data[key] = pd.Series(dim_list)

    return X_nested


def from_multi_index_to_3d_numpy(X, instance_index=None, time_index=None):
    """Convert panel data stored as pandas multi-index DataFrame to
    Numpy 3-dimensional NumPy array (n_instances, n_columns, n_timepoints).

    Parameters
    ----------
    X : pd.DataFrame
        The multi-index pandas DataFrame

    instance_index : str
        Name of the multi-index level corresponding to the DataFrame's instances

    time_index : str
        Name of multi-index level corresponding to DataFrame's timepoints

    Returns
    -------
    X_3d : np.ndarray
        3-dimensional NumPy array (n_instances, n_columns, n_timepoints)
    """
    if X.index.nlevels != 2:
        raise ValueError("Multi-index DataFrame should have 2 levels.")

    if (instance_index is None) or (time_index is None):
        msg = "Must supply parameters instance_index and time_index"
        raise ValueError(msg)

    n_instances = len(X.groupby(level=instance_index))
    # Alternative approach is more verbose
    # n_instances = (multi_ind_dataframe
    #                    .index
    #                    .get_level_values(instance_index)
    #                    .unique()).shape[0]
    n_timepoints = len(X.groupby(level=time_index))
    # Alternative approach is more verbose
    # n_instances = (multi_ind_dataframe
    #                    .index
    #                    .get_level_values(time_index)
    #                    .unique()).shape[0]

    n_columns = X.shape[1]

    X_3d = X.values.reshape(n_instances, n_timepoints, n_columns).swapaxes(1, 2)

    return X_3d


def from_3d_numpy_to_multi_index(
    X, instance_index=None, time_index=None, column_names=None
):
    """Convert 3-dimensional NumPy array (n_instances, n_columns, n_timepoints)
    to panel data stored as pandas multi-indexed DataFrame.

    Parameters
    ----------
    X : np.ndarray
        3-dimensional NumPy array (n_instances, n_columns, n_timepoints)

    instance_index : str
        Name of the multi-index level corresponding to the DataFrame's instances

    time_index : str
        Name of multi-index level corresponding to DataFrame's timepoints


    Returns
    -------
    X_mi : pd.DataFrame
        The multi-indexed pandas DataFrame
    """
    if X.ndim != 3:
        msg = " ".join(
            [
                "Input should be 3-dimensional NumPy array with shape",
                "(n_instances, n_columns, n_timepoints).",
            ]
        )
        raise TypeError(msg)

    n_instances, n_columns, n_timepoints = X.shape
    multi_index = pd.MultiIndex.from_product(
        [range(n_instances), range(n_columns), range(n_timepoints)],
        names=["instances", "columns", "timepoints"],
    )

    X_mi = pd.DataFrame({"X": X.flatten()}, index=multi_index)
    X_mi = X_mi.unstack(level="columns")

    # Assign column names
    if column_names is None:
        X_mi.columns = _make_column_names(n_columns)

    else:
        X_mi.columns = column_names

    index_rename_dict = {}
    if instance_index is not None:
        index_rename_dict["instances"] = instance_index

    if time_index is not None:
        index_rename_dict["timepoints"] = time_index

    if len(index_rename_dict) > 0:
        X_mi = X_mi.rename_axis(index=index_rename_dict)

    return X_mi


def from_multi_index_to_nested(
    multi_ind_dataframe, instance_index=None, cells_as_numpy=False
):
    """Converts a pandas DataFrame witha multi-index to a nested DataFrame

    Parameters
    ----------
    multi_ind_dataframe : pd.DataFrame
        Input multi-indexed pandas DataFrame

    instance_index_name : str
        The name of multi-index level corresponding to the DataFrame's instances

    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series


    Returns
    -------
    x_nested : pd.DataFrame
        The nested version of the DataFrame
    """
    if instance_index is None:
        raise ValueError("Supply a value for the instance_index parameter.")

    # get number of distinct cases (note: a case may have 1 or many dimensions)
    instance_idxs = multi_ind_dataframe.index.get_level_values(instance_index).unique()

    x_nested = pd.DataFrame()

    # Loop the dimensions (columns) of multi-index DataFrame
    for _label, _series in multi_ind_dataframe.iteritems():  # noqa
        # for _label in multi_ind_dataframe.columns:
        #    _series = multi_ind_dataframe.loc[:, _label]
        # Slice along the instance dimension to return list of series for each case
        # Note: if you omit .rename_axis the returned DataFrame
        #       prints time axis dimension at the start of each cell,
        #       but this doesn't affect the values.
        if cells_as_numpy:
            dim_list = [
                _series.xs(instance_idx, level=instance_index).values
                for instance_idx in instance_idxs
            ]
        else:
            dim_list = [
                _series.xs(instance_idx, level=instance_index).rename_axis(None)
                for instance_idx in instance_idxs
            ]

        x_nested[_label] = pd.Series(dim_list)
    x_nested = pd.DataFrame(x_nested)

    col_msg = "Multi-index and nested DataFrames should have same columns names"
    assert (x_nested.columns == multi_ind_dataframe.columns).all(), col_msg

    return x_nested


def from_nested_to_multi_index(X, instance_index=None, time_index=None):
    """Converts nested pandas DataFrame (with time series as pandas Series
    or NumPy array in cells) into multi-indexed pandas DataFrame.

    Can convert mixed nested and primitive DataFrame to multi-index DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        The nested DataFrame to convert to a multi-indexed pandas DataFrame

    instance_index : str
        Name of the multi-index level corresponding to the DataFrame's instances

    time_index : str
        Name of multi-index level corresponding to DataFrame's timepoints

    Returns
    -------
    X_mi : pd.DataFrame
        The multi-indexed pandas DataFrame

    """
    if not is_nested_dataframe(X):
        raise ValueError("Input DataFrame is not a nested DataFrame")

    if time_index is None:
        time_index_name = "timepoints"
    else:
        time_index_name = time_index

    # n_columns = X.shape[1]
    nested_col_mask = [*are_columns_nested(X)]

    if instance_index is None:
        instance_idxs = X.index.get_level_values(-1).unique()
        # n_instances = instance_idxs.shape[0]
        instance_index_name = "instance"

    else:
        if instance_index in X.index.names:
            instance_idxs = X.index.get_level_values(instance_index).unique()
        else:
            instance_idxs = X.index.get_level_values(-1).unique()
        # n_instances = instance_idxs.shape[0]
        instance_index_name = instance_index

    instances = []
    for instance_idx in instance_idxs:
        instance = [
            _val if isinstance(_val, pd.Series) else pd.Series(_val, name=_lab)
            for _lab, _val in X.loc[instance_idx, :].iteritems()  # noqa
        ]
        # instance = [
        #     X.loc[instance_idx, _label]
        #     if isinstance(X.loc[instance_idx, _label], pd.Series)
        #     else pd.Series(X.loc[instance_idx, _label], name=_label)
        #     for _label in X.columns ]

        instance = pd.concat(instance, axis=1)
        # For primitive (non-nested column) assume the same
        # primitive value applies to every timepoint of the instance
        for col_idx, is_nested in enumerate(nested_col_mask):
            if not is_nested:
                instance.iloc[:, col_idx] = instance.iloc[:, col_idx].ffill()

        # Correctly assign multi-index
        multi_index = pd.MultiIndex.from_product(
            [[instance_idx], instance.index],
            names=[instance_index_name, time_index_name],
        )
        instance.index = multi_index
        instances.append(instance)

    X_mi = pd.concat(instances)
    X_mi.columns = X.columns

    return X_mi


def _convert_series_cell_to_numpy(cell):
    if isinstance(cell, pd.Series):
        return cell.to_numpy()
    else:
        return cell


def from_nested_to_3d_numpy(X):
    """Convert nested pandas DataFrame (with time series as pandas Series
    in cells) into NumPy ndarray with shape
    (n_instances, n_columns, n_timepoints).

    Parameters
    ----------
    X : pd.DataFrame
        Nested pandas DataFrame

    Returns
    -------
    X_3d : np.ndarrray
        3-dimensional NumPy array
    """
    # n_instances, n_columns = X.shape
    # n_timepoints = X.iloc[0, 0].shape[0]
    # array = np.empty((n_instances, n_columns, n_timepoints))
    # for column in range(n_columns):
    #     array[:, column, :] = X.iloc[:, column].tolist()
    # return array
    if not is_nested_dataframe(X):
        raise ValueError("Input DataFrame is not a nested DataFrame")

    # n_columns = X.shape[1]
    nested_col_mask = [*are_columns_nested(X)]

    # If all the columns are nested in structure
    if nested_col_mask.count(True) == len(nested_col_mask):
        X_3d = np.stack(
            X.applymap(_convert_series_cell_to_numpy)
            .apply(lambda row: np.stack(row), axis=1)
            .to_numpy()
        )

    # If some columns are primitive (non-nested) then first convert to
    # multi-indexed DataFrame where the same value of these columns is
    # repeated for each timepoint
    # Then the multi-indexed DataFrame can be converted to 3d NumPy array
    else:
        X_mi = from_nested_to_multi_index(X)
        X_3d = from_multi_index_to_3d_numpy(
            X_mi, instance_index="instance", time_index="timepoints"
        )

    return X_3d


def from_3d_numpy_to_nested(X, column_names=None, cells_as_numpy=False):
    """Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints)
    into nested pandas DataFrame (with time series as pandas Series in cells)

    Parameters
    ----------
    X : np.ndarray
        3-dimensional Numpy array to convert to nested pandas DataFrame format

    column_names: list-like, default = None
        Optional list of names to use for naming nested DataFrame's columns

    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series


    Returns
    -------
    df : pd.DataFrame
    """
    df = pd.DataFrame()
    # n_instances, n_variables, _ = X.shape
    n_instances, n_columns, n_timepoints = X.shape

    container = np.array if cells_as_numpy else pd.Series

    if column_names is None:
        column_names = _make_column_names(n_columns)

    else:
        if len(column_names) != n_columns:
            msg = " ".join(
                [
                    f"Input 3d Numpy array as {n_columns} columns,",
                    f"but only {len(column_names)} names supplied",
                ]
            )
            raise ValueError(msg)

    for j, column in enumerate(column_names):
        df[column] = [container(X[instance, j, :]) for instance in range(n_instances)]
    return df


def is_nested_dataframe(X):
    """Checks whether the input is a nested DataFrame.

    To allow for a mixture of nested and primitive columns types the
    the considers whether any column is a nested np.ndarray or pd.Series.

    Column is consider nested if any cells in column have a nested structure.

    Parameters
    -----------
    X :
        Input that is checked to determine if it is a nested DataFrame.

    Returns
    -------
    bool :
        Whether the input is a nested DataFrame
    """
    # return isinstance(X, pd.DataFrame) and isinstance(
    #     X.iloc[0, 0], (np.ndarray, pd.Series)
    # )
    is_dataframe = isinstance(X, pd.DataFrame)

    # If not a DataFrame we know is_nested_dataframe is False
    if not is_dataframe:
        return is_dataframe

    # Otherwise we'll see if any column has a nested structure in first row
    else:
        is_nested = are_columns_nested(X).any()

        return is_dataframe and is_nested
