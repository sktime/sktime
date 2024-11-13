# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common utilities for polars based data containers."""

from sktime.datatypes._base._common import _req
from sktime.datatypes._base._common import _ret as ret
from sktime.datatypes._dtypekind import _get_feature_kind, _polars_dtype_to_kind


def get_mi_cols(obj):
    """Get multiindex or index cols from a polars object."""
    return [x for x in obj.columns if isinstance(x, str) and x.startswith("__index__")]


def is_monotonically_increasing(obj):
    """Check is polars frame columns(__index__) is monotonically increasing."""
    index_cols = get_mi_cols(obj)

    # Series with no index columns
    if len(index_cols) == 0:
        return True

    # check for Series scitype
    if len(index_cols) == 1:
        if obj[index_cols[0]].is_sorted():
            return True

    # check for Panel and Hierarchical scitype
    else:
        import polars as pl

        index_df = obj.with_columns(index_cols)
        grouped = index_df.group_by(index_cols[:-1]).agg([pl.col(index_cols[-1])])
        last_index_col = grouped.select([index_cols[-1]])
        for val in last_index_col.iter_rows():
            # iter rows returns a list of tuples
            if not pl.Series(val[0]).is_sorted():
                return False
        return True

    return False


def _convert_period_index_to_datetime_index(obj):
    """Convert PeriodIndex to DatatimeIndex as polars only supports DatetimeIndex."""
    import pandas as pd

    if isinstance(obj.index, pd.PeriodIndex):
        obj.index = obj.index.to_timestamp(freq=obj.index.freq)

    if isinstance(obj.index, pd.MultiIndex):
        levels = obj.index.levels
        if isinstance(levels[-1], pd.PeriodIndex):
            new_levels = list(levels)
            new_levels[-1] = new_levels[-1].to_timestamp(freq=new_levels[-1].freq)
            obj.index = obj.index.set_levels(new_levels)

    return obj


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

    obj = obj.copy()
    obj = _convert_period_index_to_datetime_index(
        obj
    )  # Polars only supports DatetimeIndex
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


def convert_polars_to_pandas(obj, infer_freq=True):
    """Convert polars DataFrame to pandas DataFrame, preserving MultiIndex.

    Parameters
    ----------
    obj : polars.DataFrame, polars.LazyFrame
    infer_freq : bool, optional (default=True)
        Infer frequency and set freq attribute of DatetimeIndex and DatetimeIndex levels

    Returns
    -------
    pandas.DataFrame
        MultiIndex levels are preserved in the columns of the returned DataFrame
        by converting columns with names __index__[indexname] to MultiIndex levels,
        and other columns identical to those of obj.
    """
    import pandas as pd
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

    def set_freq(obj):
        if isinstance(obj.index, pd.DatetimeIndex):
            obj.index.freq = pd.infer_freq(obj.index)

        if isinstance(obj.index, pd.MultiIndex):
            levels = obj.index.levels
            if isinstance(levels[-1], pd.DatetimeIndex):
                obj.index.levels[-1].freq = pd.infer_freq(obj.index.levels[-1])
        return obj

    pd_index_names = get_index_names(obj.columns)

    if len(pd_index_names) > 0:
        pl_index_names = get_mi_cols(obj)
        obj = obj.set_index(pl_index_names)
        obj.index.names = pd_index_names
        if infer_freq:
            obj = set_freq(obj)

    return obj


def check_polars_frame(
    obj, return_metadata=False, var_name="obj", lazy=False, scitype="Table"
):
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
    if scitype not in ["Table", "Series", "Panel", "Hierarchical"]:
        return RuntimeError(
            'scitype arg of check_polars_frame must be one of strings "Table", '
            f'"Series", "Panel", or "Hierarchical", but found {scitype}'
        )

    index_cols = []

    if scitype in ["Series", "Panel", "Hierarchical"]:
        index_cols = get_mi_cols(obj)
        n_vars = len(index_cols)
        scitypes_index = {
            "Series": n_vars == 1 or n_vars == 0,
            "Panel": n_vars == 2,
            "Hierarchical": n_vars >= 3,
        }

        if not scitypes_index[scitype]:
            cols_msg = (
                f"{var_name} must have correct number of index columns for scitype, "
                f"Series: 0 or 1, Panel: 2, Hierarchical: >= 3,"
                f"found {len(index_cols)}, namely: {index_cols}"
            )
            return ret(False, cols_msg, None, return_metadata)

        # check if index columns are monotonically increasing
        if isinstance(obj, pl.DataFrame) and not is_monotonically_increasing(obj):
            msg = (
                f"The (time) index of {var_name} must be sorted monotonically "
                f"increasing. Use {var_name}.sort() on columns representing "
                f"index(__index__) to sort the index, or {var_name}.is_duplicated() "
                f"to find duplicates."
            )
            return ret(False, msg, None, return_metadata)

    # columns in polars are unique, no check required

    if lazy:
        width = obj.collect_schema().len()
        columns = obj.collect_schema().names()
        dtypes = obj.collect_schema().dtypes()
    else:
        width = obj.width
        columns = obj.columns
        dtypes = obj.dtypes

    if _req("is_empty", return_metadata):
        metadata["is_empty"] = width < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = width - len(index_cols) == 1
    if _req("n_features", return_metadata):
        metadata["n_features"] = width - len(index_cols)
    if _req("feature_names", return_metadata):
        feature_columns = [x for x in columns if x not in index_cols]
        metadata["feature_names"] = feature_columns
    if _req("dtypekind_dfip", return_metadata):
        index_cols_count = len(index_cols)
        dtype_list = dtypes[index_cols_count:]
        metadata["dtypekind_dfip"] = _polars_dtype_to_kind(dtype_list)
    if _req("feature_kind", return_metadata):
        index_cols_count = len(index_cols)
        dtype_list = dtypes[index_cols_count:]
        dtype_kind = _polars_dtype_to_kind(dtype_list)
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

    if scitype == "Table":
        if _req("n_instances", return_metadata):
            if hasattr(obj, "height"):
                metadata["n_instances"] = obj.height
            else:
                metadata["n_instances"] = "NA"

    # check if there are any nans
    #   compute only if needed
    if _req("has_nans", return_metadata):
        if isinstance(obj, pl.LazyFrame):
            metadata["has_nans"] = "NA"
        else:
            hasnan = obj.null_count().sum_horizontal().to_numpy()[0] > 0
            metadata["has_nans"] = hasnan

    return ret(True, None, metadata, return_metadata)
