# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Converter utilities between dask and pandas, with multiindex convention.

Converts between:
pd.DataFrames with ordinary (single-level) index or pd.Multiindex, and
dask DataFrame

If pd.DataFrame has ordinary index, converts using dask compute/from_pandas

if pd.DataFrame has MultiIndex, converts and back-converts
MultiIndex columns 0:-1 to DataFrame columns with the name:
    __index__[indexname], if level has a name indexname
    __index__[index_iloc], if level has no indexname and is index_iloc-th level
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
        obj = obj.set_index(multi_cols, append=True)
        nlevels = len(obj.index.names)
        order = list(range(1, nlevels)) + [0]
        obj = obj.reorder_levels(order)

        names = obj.index.names[:-1]
        new_names = mi_names(names)
        # names = [mi_name(x) for x in names]
        new_names = new_names + [obj.index.names[-1]]

        obj.index.names = new_names

    return obj


def convert_pandas_to_dask(obj):
    """Convert pandas DataFrame to dask DataFrame, preserving MultiIndex.

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
    from dask.dataframe import from_pandas

    def dask_mi_names(names):
        res = list(names).copy()
        for i, name in enumerate(names):
            if name is None:
                res[i] = str(i)
        return [f"__index__{x}" for x in res]

    if isinstance(obj.index, pd.MultiIndex):
        names = obj.index.names[:-1]
        new_names = dask_mi_names(names) + [obj.index.names[-1]]
        n_index = len(names)

        obj = obj.copy()
        obj.index.names = new_names
        obj = obj.reset_index(level=list(range(n_index)))

    obj = from_pandas(obj, npartitions=1)

    return obj
