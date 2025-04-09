"""Coercion utilities for mtypes."""

__author__ = ["fkiraly"]

import numbers

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


def _convert_variable_name(name, i: int, prefix: str, strategy: str):
    if strategy == "all":
        return prefix + "var" + str(i)
    elif strategy == "none":
        return (prefix + "var" + str(i)) if name is None else name
    elif strategy == "integer":
        return (prefix + "var" + str(i)) if isinstance(name, numbers.Integral) else name
    elif strategy == "none_integer":
        return (
            (prefix + "var" + str(i))
            if (name is None or isinstance(name, numbers.Integral))
            else name
        )
    else:
        raise ValueError(
            "strategy should be one of [all, none, integer, none_integer]"
            f" but found {strategy}"
        )


def _coerce_variable_name(obj, prefix="", strategy: str = "all"):
    if isinstance(obj, pd.Series):
        old_names = [obj.name]
        obj.name = _convert_variable_name(obj.name, 0, prefix, strategy)
        new_names = [obj.name]
    elif isinstance(obj, pd.DataFrame):
        old_names = list(obj.columns)
        obj.columns = [
            _convert_variable_name(obj.columns[i], i, prefix, strategy)
            for i in range(len(obj.columns))
        ]
        new_names = list(obj.columns)
    else:
        return obj, [], []
    return obj, old_names, new_names


def _restore_variable_name(obj, old_names: list, new_names: list):
    if isinstance(obj, pd.Series):
        assert obj.name == new_names[0], (
            f"Series name: {obj.name} not match with new_names: {new_names}"
        )
        assert len(old_names) == 1, "len(old_names) must equal to 1 for series"
        obj.name = old_names[0]
    elif isinstance(obj, pd.DataFrame):
        assert list(obj.columns) == new_names, (
            f"DataFrame name(s): {obj.columns} not match with new_names: {new_names}"
        )
        obj.columns = old_names

    return obj
