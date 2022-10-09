# -*- coding: utf-8 -*-
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


def convert_dask_to_pandas(obj):
    """Convert dask DataFrame to pandas DataFrame, preserving MultiIndex.

    Parameters
    ----------
    obj : pandas.DataFrame

    Returns
    -------
    dask DataFrame
        MultiIndex levels 0 .. -1 of X are converted to columns of name
        __index__[indexname], where indexname is name of multiindex level,
        or the integer index if the level has no name
        other columns and column names are identical to those of X
    """
    obj = obj.compute()

    def is_mi_col(x):
        return isinstance(x, str) and x.startswith("__index__")

    def mi_name(x):
        return x.split("__index__")[1]

    def mi_names(names):
        new_names = [mi_name(x) for x in names]
        for i, name in enumerate(new_names):
            if name == str(i):
                new_names[i] = None
        return new_names

    multi_cols = [x for x in obj.columns if is_mi_col(x)]

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
    obj : dask DataFrame
    npartitions : int or None, optional, default = 1
        npartitions passed to dask from_pandas when converting obj to dask
    chunksize : int or None, optional, default = None
        chunksize passed to dask from_pandas when converting obj to dask
    sort : bool, optional, default = True
        sort passed to dask from_pandas when converting obj to dask

    Returns
    -------
    pandas.DataFrame
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
