# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common utilities for polars based data containers."""

from sktime.datatypes._common import _req
from sktime.datatypes._common import _ret as ret


def get_mi_cols(obj):
    """Get multiindex or index cols from a polars object."""
    return [x for x in obj.columns if isinstance(x, str) and x.startswith("__index__")]


def convert_pandas_to_polars(
    obj, schema_overrides=None, rechunk=True, nan_to_null=True, lazy=False
):
    """Convert pandas DataFrame to polars DataFrame. Preserving MultiIndex.

    Parameters
    ----------
    obj : pandas.DataFrame
    schema_overrides : dict, optional (default=None)
        Support override of inferred types for one or more columns.
    rechunk : bool, optional (default=True)
        Make sure that all data is in contiguous memory.
    nan_to_null : bool, optional (default=True)
        If data contains NaN values PyArrow will convert the NaN to None
    lazy : bool, optional (default=False)
        If True, return a LazyFrame instead of a DataFrame

    Returns
    -------
    polars.DataFrame or polars.LazyFrame (if lazy=True)
        MultiIndex levels 0 .. -1 of X are converted to columns of name
        __index__[indexname], where indexname is name of multiindex level,
        or the integer index if the level has no name
        other columns and column names are identical to those of X
    """
    from polars import from_pandas

    def polars_index_columns(names):
        col = names.copy()
        for i, name in enumerate(names):
            if name is None:
                col[i] = f"__index__{i}"
            else:
                col[i] = f"__index__{name}"
        return col

    index_names = obj.index.names
    index_names = polars_index_columns(index_names)
    obj.index.names = index_names
    obj = obj.reset_index()

    obj = from_pandas(
        obj, schema_overrides=schema_overrides, rechunk=rechunk, nan_to_null=nan_to_null
    )

    if lazy:
        obj = obj.lazy()

    return obj


def convert_polars_to_pandas(obj):
    """Convert polars DataFrame to pandas DataFrame, preserving MultiIndex.

    Parameters
    ----------
    obj : polars.DataFrame, polars.LazyFrame

    Returns
    -------
    pandas.DataFrame
        MultiIndex levels are preserved in the columns of the returned DataFrame
        by converting columns with names __index__[indexname] to MultiIndex levels,
        and other columns identical to those of obj.
    """
    from polars.lazyframe.frame import LazyFrame

    # convert to DataFrame if LazyFrame
    if isinstance(obj, LazyFrame):
        obj = obj.collect()

    obj = obj.to_pandas()

    def index_name(x):
        return x.split("__index__")[1]

    def get_index_names(cols):
        names = [index_name(x) for x in cols if x.startswith("__index__")]
        for i, name in enumerate(names):
            if name == str(i):
                names[i] = None
        return names

    pd_index_names = get_index_names(obj.columns)

    if len(pd_index_names) > 0:
        pl_index_names = get_mi_cols(obj)
        obj = obj.set_index(pl_index_names)
        obj.index.names = pd_index_names

    return obj


def check_polars_frame(obj, return_metadata=False, var_name="obj", lazy=False):
    """Check polars frame, generic format."""
    import polars as pl

    metadata = {}

    if lazy:
        exp_type = pl.LazyFrame
        exp_type_str = "LazyFrame"
    else:
        exp_type = pl.DataFrame
        exp_type_str = "DataFrame"

    if not isinstance(obj, exp_type):
        msg = f"{var_name} must be a polars {exp_type_str}, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a polars DataFrame or LazyFrame
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = obj.width < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = obj.width == 1
    if _req("n_instances", return_metadata):
        if hasattr(obj, "height"):
            metadata["n_instances"] = obj.height
        else:
            metadata["n_instances"] = "NA"
    if _req("n_features", return_metadata):
        metadata["n_features"] = obj.width
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns

    # check if there are any nans
    #   compute only if needed
    if _req("has_nans", return_metadata):
        if isinstance(obj, pl.LazyFrame):
            metadata["has_nans"] = "NA"
        else:
            hasnan = obj.null_count().sum_horizontal().to_numpy()[0] > 0
            metadata["has_nans"] = hasnan

    return ret(True, None, metadata, return_metadata)
