# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Converter utilities between dask and pandas, with multiindex convention.

Converts between:
pd.DataFrames with ordinary (single-level) index or pd.Multiindex, and
dask DataFrame

If pd.DataFrame has ordinary index, converts using dask compute/from_pandas

if pd.DataFrame has MultiIndex, converts and back-converts
MultiIndex columns to DataFrame columns with the name:
    __index__[indexname], if level has a name indexname
    __index__[index_iloc], if level has no indexname and is index_iloc-th level
index is replaced by a string index where tuples are replaced with str coerced elements
"""

import pandas as pd

from sktime.datatypes._common import _req
from sktime.datatypes._common import _ret as ret
from sktime.datatypes._dtypekind import _get_feature_kind, _pandas_dtype_to_kind


def _is_mi_col(x):
    return isinstance(x, str) and x.startswith("__index__")


def get_mi_cols(obj):
    """Get multiindex cols from a dask object.

    Parameters
    ----------
    obj : dask DataFrame

    Returns
    -------
    list of pandas index elements
        all column index elements of obj that start with __index__
        i.e., columns that are interpreted as multiindex columns  in the correspondence
    """
    return [x for x in obj.columns if _is_mi_col(x)]


def convert_dask_to_pandas(obj):
    """Convert dask DataFrame to pandas DataFrame, preserving MultiIndex.

    Parameters
    ----------
    obj : dask DataFrame

    Returns
    -------
    pandas.DataFrame
        MultiIndex levels 0 .. -1 of X are converted to columns of name
        __index__[indexname], where indexname is name of multiindex level,
        or the integer index if the level has no name
        other columns and column names are identical to those of X
    """
    obj = obj.compute()

    def mi_name(x):
        return x.split("__index__")[1]

    def mi_names(names):
        new_names = [mi_name(x) for x in names]
        for i, name in enumerate(new_names):
            if name == str(i):
                new_names[i] = None
        return new_names

    multi_cols = get_mi_cols(obj)

    # if has multi-index cols, move to pandas MultiIndex
    if len(multi_cols) > 0:
        obj = obj.set_index(multi_cols)

        names = obj.index.names
        new_names = mi_names(names)
        new_names = new_names

        obj.index.names = new_names

    return obj


def convert_pandas_to_dask(obj, npartitions=1, chunksize=None, sort=True):
    """Convert pandas DataFrame to dask DataFrame, preserving MultiIndex.

    Parameters
    ----------
    obj : pandas.DataFrame
    npartitions : int or None, optional, default = 1
        npartitions passed to dask from_pandas when converting obj to dask
    chunksize : int or None, optional, default = None
        chunksize passed to dask from_pandas when converting obj to dask
    sort : bool, optional, default = True
        sort passed to dask from_pandas when converting obj to dask

    Returns
    -------
    dask DataFrame
        MultiIndex levels 0 .. -1 of X are converted to columns of name
        __index__[indexname], where indexname is name of multiindex level,
        or the integer index if the level has no name
        other columns and column names are identical to those of X
    """
    from dask.dataframe import from_pandas

    def dask_mi_names(names):
        res = list(names).copy()
        for i, name in enumerate(names):
            if name is None:
                res[i] = str(i)
        return [f"__index__{x}" for x in res]

    if isinstance(obj.index, pd.MultiIndex):
        names = obj.index.names
        new_names = dask_mi_names(names)
        new_index = [str(x) for x in obj.index]

        obj = obj.copy()
        obj.index.names = new_names
        obj = obj.reset_index()
        obj.index = new_index

    obj = from_pandas(obj, npartitions=npartitions, chunksize=chunksize, sort=sort)

    return obj


def check_dask_frame(
    obj, return_metadata=False, var_name="obj", freq_set_check=False, scitype="Series"
):
    """Check dask frame, generic for sktime check format."""
    import dask

    metadata = {}

    if not isinstance(obj, dask.dataframe.core.DataFrame):
        msg = f"{var_name} must be a dask DataFrame, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a dask DataFrame

    index_cols = get_mi_cols(obj)

    # check right number of cols depending on scitype
    if scitype == "Series":
        cols_msg = (
            f"{var_name} must have exactly one index column, "
            f"found {len(index_cols)}, namely: {index_cols}"
        )
        right_no_index_cols = len(index_cols) <= 1
    elif scitype == "Panel":
        cols_msg = (
            f"{var_name} must have exactly two index columns, "
            f"found {len(index_cols)}, namely: {index_cols}"
        )
        right_no_index_cols = len(index_cols) == 2
    elif scitype == "Hierarchical":
        cols_msg = (
            f"{var_name} must have three or more index columns, "
            f"found {len(index_cols)}, namely: {index_cols}"
        )
        right_no_index_cols = len(index_cols) >= 3
    else:
        return RuntimeError(
            'scitype arg of check_dask_frame must be one of strings "Series", '
            f'"Panel", or "Hierarchical", but found {scitype}'
        )

    if not right_no_index_cols:
        # dask series should have at most one __index__ col
        return ret(False, cols_msg, None, return_metadata)

    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj.index) < 1 or len(obj.columns) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = len(obj.columns) == 1
    if _req("n_features", return_metadata):
        metadata["n_features"] = len(obj.columns)
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns.to_list()
    if _req("dtypekind_dfip", return_metadata):
        index_cols_count = len(index_cols)
        # slicing off additional index columns
        dtype_list = obj.dtypes.to_list()[index_cols_count:]
        metadata["dtypekind_dfip"] = _pandas_dtype_to_kind(dtype_list)
    if _req("feature_kind", return_metadata):
        index_cols_count = len(index_cols)
        dtype_list = obj.dtypes.to_list()[index_cols_count:]
        dtype_kind = _pandas_dtype_to_kind(dtype_list)
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    # check that columns are unique
    if not obj.columns.is_unique:
        msg = f"{var_name} must have unique column indices, but found {obj.columns}"
        return ret(False, msg, None, return_metadata)

    # check whether the time index is of valid type
    # if not is_in_valid_index_types(index):
    #     msg = (
    #         f"{type(index)} is not supported for {var_name}, use "
    #         f"one of {VALID_INDEX_TYPES} or integer index instead."
    #     )
    #     return ret(False, msg, None, return_metadata)

    # Check time index is ordered in time
    if not obj.index.is_monotonic_increasing.compute():
        msg = (
            f"The (time) index of {var_name} must be sorted "
            f"monotonically increasing, but found: {obj.index}"
        )
        return ret(False, msg, None, return_metadata)

    if freq_set_check and isinstance(obj.index, pd.DatetimeIndex):
        if obj.index.freq is None:
            msg = f"{var_name} has DatetimeIndex, but no freq attribute set."
            return ret(False, msg, None, return_metadata)

    # check whether index is equally spaced or if there are any nans
    #   compute only if needed
    if _req("is_equally_spaced", return_metadata):
        # todo: logic for equal spacing
        metadata["is_equally_spaced"] = True
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isnull().values.any().compute()

    if scitype in ["Panel", "Hierarchical"]:
        if _req("n_instances", return_metadata):
            instance_cols = index_cols[:-1]
            metadata["n_instances"] = len(obj[instance_cols].drop_duplicates())

    if scitype in ["Hierarchical"]:
        if _req("n_panels", return_metadata):
            panel_cols = index_cols[:-2]
            metadata["n_panels"] = len(obj[panel_cols].drop_duplicates())

    return ret(True, None, metadata, return_metadata)
