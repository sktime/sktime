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
            obj = obj.astype({col: "float" for col in nullable_cols})
        return obj

    return obj


def _coerce_variable_name(obj: pd.Series | pd.DataFrame, prefix=""):
    if isinstance(obj, pd.Series):
        old_names = [obj.name]
        obj.name = prefix + "var0"
        new_names = [obj.name]
    elif isinstance(obj, pd.DataFrame):
        old_names = list(obj.columns)
        obj.columns = [prefix + "var" + str(i) for i in range(len(obj.columns))]
        new_names = list(obj.columns)
    else:
        return obj, [], []
    return obj, old_names, new_names


def _restore_variable_name(
    obj: pd.Series | pd.DataFrame, old_names: list, new_names: list
):
    if isinstance(obj, pd.Series):
        assert (
            obj.name == new_names[0]
        ), f"Series name: {obj.name} not match with new_names: {new_names}"
        assert len(old_names) == 1, "len(old_names) must equal to 1 for series"
        obj.name = old_names[0]
    elif isinstance(obj, pd.DataFrame):
        assert (
            list(obj.columns) == new_names
        ), f"DataFrame name(s): {obj.columns} not match with new_names: {new_names}"
        obj.columns = old_names

    return obj
