# -*- coding: utf-8 -*-
"""Machine type converters for Panel scitype.

Exports conversion and mtype dictionary for Panel scitype:

convert_dict: dict indexed by triples of str
  1st element = convert from - str
  2nd element = convert to - str
  3rd element = considered as this scitype - str
elements are conversion functions of machine type (1st) -> 2nd

Function signature of all elements
convert_dict[(from_type, to_type, as_scitype)]

Parameters
----------
obj : from_type - object to convert
store : dictionary - reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

Raises
------
ValueError and TypeError, if requested conversion is not possible
                            (depending on conversion logic)
"""

import numpy as np
import pandas as pd

__all__ = [
    "convert_dict",
]

from sktime.datatypes._panel._registry import MTYPE_LIST_PANEL

# dictionary indexed by triples of types
#  1st element = convert from - type
#  2nd element = convert to - type
#  3rd element = considered as this scitype - string
# elements are conversion functions of machine type (1st) -> 2nd

convert_dict = dict()


def convert_identity(obj, store=None):

    return obj


# assign identity function to type conversion to self
for tp in MTYPE_LIST_PANEL:
    convert_dict[(tp, tp, "Panel")] = convert_identity


def _cell_is_series_or_array(cell):
    return isinstance(cell, (pd.Series, np.ndarray))


def _nested_cell_mask(X):
    return X.applymap(_cell_is_series_or_array)


def are_columns_nested(X):
    """Check whether any cells have nested structure in each DataFrame column.

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
    """Check if all time-series in nested pandas DataFrame have the same index.

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
    """Convert 2D NumPy Panel to 2D numpy Panel.

    Converts 3D numpy array (n_instances, n_columns, n_timepoints) to
    a 2D numpy array with shape (n_instances, n_columns*n_timepoints)

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


def from_3d_numpy_to_2d_array_adp(obj, store=None):

    return from_3d_numpy_to_2d_array(obj)


convert_dict[("numpy3D", "numpyflat", "Panel")] = from_3d_numpy_to_2d_array_adp


def from_nested_to_2d_array(X, return_numpy=False):
    """Convert nested Panel to 2D numpy Panel.

    Convert nested pandas DataFrame or Series with NumPy arrays or
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


def from_nested_to_pdwide(obj, store=None):

    return from_nested_to_2d_array(X=obj, return_numpy=False)


def from_nested_to_2d_np_array(obj, store=None):

    return from_nested_to_2d_array(X=obj, return_numpy=True)


convert_dict[("nested_univ", "pd-wide", "Panel")] = from_nested_to_pdwide

convert_dict[("nested_univ", "numpyflat", "Panel")] = from_nested_to_2d_np_array


def from_2d_array_to_nested(
    X, index=None, columns=None, time_index=None, cells_as_numpy=False
):
    """Convert 2D dataframe to nested dataframe.

    Convert tabular pandas DataFrame with only primitives in cells into
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


def from_pd_wide_to_nested(obj, store=None):

    return from_2d_array_to_nested(X=obj)


convert_dict[("pd-wide", "nested_univ", "Panel")] = from_pd_wide_to_nested


def convert_from_dictionary(ts_dict):
    """Conversion from dictionary to pandas.

    Simple conversion from a dictionary of each series, e.g. univariate
        x = {
            "Series1": [1.0,2.0,3.0,1.0,2.0],
            "Series2": [3.0,2.0,1.0,3.0,2.0],
        }
    or multivariate, e.g.
    to sktime pandas format
    TODO: Adapt for multivariate
    """
    panda = pd.DataFrame(ts_dict)
    panda = panda.transpose()
    return from_2d_array_to_nested(panda)


def _concat_nested_arrays(arrs, cells_as_numpy=False):
    """Nest tabular arrays from nested list.

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
    """Get index of time series data, helper function.

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


def from_nested_to_long_adp(obj, store=None):

    return from_nested_to_long(
        X=obj,
        instance_column_name="case_id",
        time_column_name="reading_id",
        dimension_column_name="dim_id",
    )


convert_dict[("nested_univ", "pd-long", "Panel")] = from_nested_to_long_adp


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

    return X_nested


def from_long_to_nested_adp(obj, store=None):

    return from_long_to_nested(X_long=obj)


convert_dict[("pd-long", "nested_univ", "Panel")] = from_nested_to_long_adp


def from_multi_index_to_3d_numpy(X):
    """Convert pandas multi-index Panel to numpy 3D Panel.

    Convert panel data stored as pandas multi-index DataFrame to
    Numpy 3-dimensional NumPy array (n_instances, n_columns, n_timepoints).

    Parameters
    ----------
    X : pd.DataFrame
        The multi-index pandas DataFrame

    Returns
    -------
    X_3d : np.ndarray
        3-dimensional NumPy array (n_instances, n_columns, n_timepoints)
    """
    if X.index.nlevels != 2:
        raise ValueError("Multi-index DataFrame should have 2 levels.")

    n_instances = len(X.index.get_level_values(0).unique())
    n_timepoints = len(X.index.get_level_values(1).unique())
    n_columns = X.shape[1]

    X_3d = X.values.reshape(n_instances, n_timepoints, n_columns).swapaxes(1, 2)

    return X_3d


def from_multi_index_to_3d_numpy_adp(obj, store=None):

    res = from_multi_index_to_3d_numpy(X=obj)
    if isinstance(store, dict):
        store["columns"] = obj.columns
        store["index_names"] = obj.index.names

    return res


convert_dict[("pd-multiindex", "numpy3D", "Panel")] = from_multi_index_to_3d_numpy_adp


def from_3d_numpy_to_multi_index(
    X, instance_index=None, time_index=None, column_names=None
):
    """Convert 3D numpy Panel to pandas multi-index Panel.

    Convert 3-dimensional NumPy array (n_instances, n_columns, n_timepoints)
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


def from_3d_numpy_to_multi_index_adp(obj, store=None):
    res = from_3d_numpy_to_multi_index(X=obj)
    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == obj.shape[1]
    ):
        res.columns = store["columns"]

    if isinstance(store, dict) and "index_names" in store.keys():
        res.index.names = store["index_names"]

    return res


convert_dict[("numpy3D", "pd-multiindex", "Panel")] = from_3d_numpy_to_multi_index_adp


def from_multi_index_to_nested(
    multi_ind_dataframe, instance_index=None, cells_as_numpy=False
):
    """Convert a pandas DataFrame witha multi-index to a nested DataFrame.

    Parameters
    ----------
    multi_ind_dataframe : pd.DataFrame
        Input multi-indexed pandas DataFrame

    instance_index_name : int or str, default=0 (first level = 0-th index)
        Index or name of multi-index level corresponding to the DataFrame's instances

    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series

    Returns
    -------
    x_nested : pd.DataFrame
        The nested version of the DataFrame
    """
    if instance_index is None:
        instance_index = 0

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
    x_nested = pd.DataFrame(x_nested).set_axis(instance_idxs)

    col_msg = "Multi-index and nested DataFrames should have same columns names"
    assert (x_nested.columns == multi_ind_dataframe.columns).all(), col_msg

    return x_nested


def from_multi_index_to_nested_adp(obj, store=None):

    if isinstance(store, dict):
        store["index_names"] = obj.index.names

    return from_multi_index_to_nested(multi_ind_dataframe=obj, instance_index=None)


convert_dict[("pd-multiindex", "nested_univ", "Panel")] = from_multi_index_to_nested_adp


def from_nested_to_multi_index(X, instance_index=None, time_index=None):
    """Convert nested pandas Panel to multi-index pandas Panel.

    Converts nested pandas DataFrame (with time series as pandas Series
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
    # this contains the right values, but does not have the right index
    #   need convert_dtypes or dtypes will always be object
    # explode by column to ensure we deal with unequal length series properly
    X_mi = pd.DataFrame()

    X_cols = X.columns
    nested_cols = [c for c in X_cols if isinstance(X[[c]].iloc[0, 0], pd.Series)]
    non_nested_cols = list(set(X_cols).difference(nested_cols))

    for c in nested_cols:
        X_col = X[[c]].explode(c)
        X_col = X_col.infer_objects()

        # create the right MultiIndex and assign to X_mi
        idx_df = X[[c]].applymap(lambda x: x.index).explode(c)
        idx_df = idx_df.set_index(c, append=True)
        X_col.index = idx_df.index.set_names([instance_index, time_index])

        X_mi[[c]] = X_col

    for c in non_nested_cols:
        for ix in X.index:
            X_mi.loc[ix, c] = X[[c]].loc[ix].iloc[0]
        X_mi[[c]] = X_mi[[c]].convert_dtypes()

    return X_mi


def from_nested_to_multi_index_adp(obj, store=None):

    res = from_nested_to_multi_index(
        X=obj, instance_index="instances", time_index="timepoints"
    )

    if isinstance(store, dict) and "index_names" in store.keys():
        res.index.names = store["index_names"]

    return res


convert_dict[("nested_univ", "pd-multiindex", "Panel")] = from_nested_to_multi_index_adp


def _convert_series_cell_to_numpy(cell):
    if isinstance(cell, pd.Series):
        return cell.to_numpy()
    else:
        return cell


def from_nested_to_3d_numpy(X):
    """Convert nested Panel to 3D numpy Panel.

    Convert nested pandas DataFrame (with time series as pandas Series
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
        X_3d = from_multi_index_to_3d_numpy(X_mi)

    return X_3d


def from_nested_to_3d_numpy_adp(obj, store=None):

    return from_nested_to_3d_numpy(X=obj)


convert_dict[("nested_univ", "numpy3D", "Panel")] = from_nested_to_3d_numpy_adp


def from_3d_numpy_to_nested(X, column_names=None, cells_as_numpy=False):
    """Convert numpy 3D Panel to nested pandas Panel.

    Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints)
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
    n_instances, n_columns, n_timepoints = X.shape
    array_type = X.dtype

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

    column_list = []
    for j, column in enumerate(column_names):
        nested_column = (
            pd.DataFrame(X[:, j, :])
            .apply(lambda x: [container(x, dtype=array_type)], axis=1)
            .str[0]
            .rename(column)
        )
        column_list.append(nested_column)
    df = pd.concat(column_list, axis=1)
    return df


def from_3d_numpy_to_nested_adp(obj, store=None):

    return from_3d_numpy_to_nested(X=obj)


convert_dict[("numpy3D", "nested_univ", "Panel")] = from_3d_numpy_to_nested_adp


def from_dflist_to_multiindex(obj, store=None):

    n = len(obj)

    mi = pd.concat(obj, axis=0, keys=range(n), names=["instances", "timepoints"])

    if isinstance(store, dict) and "index_names" in store.keys():
        mi.index.names = store["index_names"]

    return mi


convert_dict[("df-list", "pd-multiindex", "Panel")] = from_dflist_to_multiindex


def from_multiindex_to_dflist(obj, store=None):

    instance_index = obj.index.levels[0]

    Xlist = [obj.loc[i].rename_axis(None) for i in instance_index]

    if isinstance(store, dict):
        store["index_names"] = obj.index.names

    return Xlist


convert_dict[("pd-multiindex", "df-list", "Panel")] = from_multiindex_to_dflist


def from_dflist_to_numpy3D(obj, store=None):

    if not isinstance(obj, list):
        raise TypeError("obj must be a list of pd.DataFrame")

    n = len(obj[0])
    cols = set(obj[0].columns)

    for i in range(len(obj)):
        if not n == len(obj[i]) or not set(obj[i].columns) == cols:
            raise ValueError("elements of obj must have same length and columns")

    nparr = np.array([X.to_numpy().transpose() for X in obj])

    return nparr


convert_dict[("df-list", "numpy3D", "Panel")] = from_dflist_to_numpy3D


def from_numpy3d_to_dflist(obj, store=None):

    if not isinstance(obj, np.ndarray) or len(obj.shape) != 3:
        raise TypeError("obj must be a 3D numpy.ndarray")

    cols = _make_column_names(obj.shape[1])
    Xlist = [pd.DataFrame(obj[i].T, columns=cols) for i in range(len(obj))]

    return Xlist


convert_dict[("numpy3D", "df-list", "Panel")] = from_numpy3d_to_dflist


def from_nested_to_df_list_adp(obj, store=None):

    # this is not already implemented, so chain two conversions
    obj = from_nested_to_multi_index_adp(obj, store=store)
    return from_multiindex_to_dflist(obj, store=store)


convert_dict[("nested_univ", "df-list", "Panel")] = from_nested_to_df_list_adp


def from_df_list_to_nested_adp(obj, store=None):

    # this is not already implemented, so chain two conversions
    obj = from_dflist_to_multiindex(obj, store=store)
    return from_multi_index_to_nested_adp(obj, store=store)


convert_dict[("df-list", "nested_univ", "Panel")] = from_df_list_to_nested_adp


def from_numpy3d_to_numpyflat(obj, store=None):

    if not isinstance(obj, np.ndarray) or len(obj.shape) != 3:
        raise TypeError("obj must be a 3D numpy.ndarray")

    shape = obj.shape

    # store second dimension shape/length if we want to restore
    if isinstance(store, dict):
        store["numpy_second_dim"] = shape[1]

    obj_in_2D = obj.reshape(shape[0], shape[1] * shape[2])

    return obj_in_2D


convert_dict[("numpy3D", "numpyflat", "Panel")] = from_numpy3d_to_numpyflat


def from_numpyflat_to_numpy3d(obj, store=None):

    if not isinstance(obj, np.ndarray) or len(obj.shape) != 2:
        raise TypeError("obj must be a 2D numpy.ndarray")

    shape = obj.shape

    # if store has old 2nd dimension, try to restore, otherwise assume 1
    if (
        isinstance(store, dict)
        and "numpy_second_dim" in store.keys()
        and isinstance(store["numpy_second_dim"], int)
        and shape[1] % store["numpy_second_dim"] == 0
    ):
        shape_1 = store["numpy_second_dim"]
        target_shape = (shape[0], shape_1, shape[1] / shape_1)
    else:
        target_shape = (shape[0], 1, shape[1])

    obj_in_3D = obj.reshape(target_shape)

    return obj_in_3D


convert_dict[("numpyflat", "numpy3D", "Panel")] = from_numpyflat_to_numpy3d


# obtain other conversions from/to numpyflat via concatenation to numpy3D
def _concat(fun1, fun2):
    def concat_fun(obj, store=None):
        obj1 = fun1(obj, store=store)
        obj2 = fun2(obj1, store=store)
        return obj2

    return concat_fun


for tp in set(MTYPE_LIST_PANEL).difference(["numpyflat", "numpy3D"]):
    if ("numpy3D", tp, "Panel") in convert_dict.keys():
        if ("numpyflat", tp, "Panel") not in convert_dict.keys():
            convert_dict[("numpyflat", tp, "Panel")] = _concat(
                convert_dict[("numpyflat", "numpy3D", "Panel")],
                convert_dict[("numpy3D", tp, "Panel")],
            )
    if (tp, "numpy3D", "Panel") in convert_dict.keys():
        if (tp, "numpyflat", "Panel") not in convert_dict.keys():
            convert_dict[(tp, "numpyflat", "Panel")] = _concat(
                convert_dict[(tp, "numpy3D", "Panel")],
                convert_dict[("numpy3D", "numpyflat", "Panel")],
            )
