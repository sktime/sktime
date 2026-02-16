"""Coercion utilities for mtypes."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd


def _is_nullable_numeric(dtype):
    return dtype in ["Int64", "Float64", "boolean"]


def _coerce_df_dtypes(obj):
    """Coerce pandas objects to non-nullable column types.

    Returns shallow copy and does not mutate input `obj`.

    Parameters
    ----------
    obj: pandas Series or DataFrame, or any object

    Returns
    -------
    obj unchanged, if obj is not pandas Series or DataFrame
    if obj is pandas Series or DataFrame,
        coerces nullable numeric columns to float (by reference via astype)
    """
    if isinstance(obj, pd.Series):
        if _is_nullable_numeric(obj.dtype):
            return obj.astype("float")
        return obj

    if isinstance(obj, pd.DataFrame):
        nullable_cols = [
            col for col in obj.columns if _is_nullable_numeric(obj.dtypes[col])
        ]
        if len(nullable_cols) > 0:
            obj = obj.astype(dict.fromkeys(nullable_cols, "float"))
        return obj

    return obj


def _coerce_multiindex_time_level_to_valid(obj):
    """Coerce MultiIndex time level (last level) to valid type per VALID_INDEX_TYPES.

    Edge case: pandas MultiIndex with time level not in VALID_INDEX_TYPES
    (e.g. float) so that downstream checks and conversions work. Used in
    self-conversion for pd_multiindex_hier and pd-multiindex.
    """
    from sktime.utils.validation.series import is_in_valid_index_types

    if not (
        isinstance(obj, pd.DataFrame)
        and isinstance(obj.index, pd.MultiIndex)
        and obj.index.nlevels >= 2
        and not is_in_valid_index_types(obj.index.levels[-1])
    ):
        return obj
    obj = obj.copy()
    lev = obj.index.levels[-1]
    arr = np.nan_to_num(
        np.asarray(lev, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.int64)
    new_level = pd.Index(arr, dtype=np.int64)
    obj.index = obj.index.set_levels(new_level, level=-1)
    return obj


def _coerce_series_index_to_valid(obj):
    """Coerce Series/DataFrame (single series) time index to valid type.

    Per VALID_INDEX_TYPES. For pd.Series or pd.DataFrame with a single index
    (time). If that index is
    not in VALID_INDEX_TYPES but is numeric, coerce to int64. Used in
    self-conversion for pd.Series and pd.DataFrame (Series scitype).

    Added for consistency with Panel/Hierarchical coercion; same edge case
    (float or other invalid-but-numeric time index). Can be reverted to
    only Panel + Hierarchical if preferred.
    """
    from sktime.utils.validation.series import is_in_valid_index_types

    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj
    index = obj.index
    if is_in_valid_index_types(index):
        return obj
    if not isinstance(index, pd.Index) or not np.issubdtype(index.dtype, np.number):
        return obj
    obj = obj.copy()
    arr = np.nan_to_num(
        np.asarray(index, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.int64)
    obj.index = pd.Index(arr, dtype=np.int64)
    return obj
