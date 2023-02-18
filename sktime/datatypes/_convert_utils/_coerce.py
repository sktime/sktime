# -*- coding: utf-8 -*-
"""Conercion utilities for mtypes."""

__author__ = ["fkiraly"]

import pandas as pd


def _is_nullable_numeric(dtype):

    return dtype in ["Int64", "Float64"]


def _coerce_df_dtypes(obj):
    """Coerce pandas objects to non-nullable column types.

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

    if isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            if _is_nullable_numeric(obj.dtypes[col]):
                obj[col] = obj[col].astype("float")
        return obj

    return obj
