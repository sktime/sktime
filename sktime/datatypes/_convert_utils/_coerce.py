"""Coercion utilities for mtypes."""

__author__ = ["fkiraly"]

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
