"""Machine type converters for Series to Panel.

Exports conversion functions for conversions between series scitypes:

convert_Series_to_Panel(obj, store=None)
    converts obj of Series mtype to "adjacent" Panel mtype (e.g., numpy to numpy)
convert_Panel_to_Series(obj, store=None)
    converts obj of Panel mtype to "adjacent" Series mtype (e.g., numpy to numpy)
convert_Series_to_Hierarchical(obj, store=None)
convert_Hierarchical_to_series(obj, store=None)
convert_Panel_to_Hierarchical(obj, store=None)
convert_Hierarchical_to_Panel(obj, store=None)
    converts to pd.DataFrame based data container in the target scitype
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.datatypes import convert_to, scitype
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("polars", severity="none"):
    import polars as pl

    from sktime.datatypes._adapter.polars import (
        get_mi_cols,
    )

    _HAS_POLARS = True
else:
    _HAS_POLARS = False


def convert_Series_to_Panel(obj, store=None, return_to_mtype=False):
    """Convert series to a single-series panel.

    Adds a dummy dimension to the series.
    For pd.Series or DataFrame, this results in a list of DataFrame (dim added is list).
    For numpy array, this results in a third dimension being added.
    For polars DataFrame/LazyFrame, this adds an instance column to obtain polars_panel.

    Assumes input is conformant with one of the Series mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Series
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    if obj was pd.Series or pd.DataFrame, returns a panel of mtype df-list
    if obj was np.ndarray, returns a panel of mtype numpy3D
    if obj was polars.DataFrame or LazyFrame, returns polars_panel
    """
    if isinstance(obj, pd.Series):
        obj = pd.DataFrame(obj)

    if isinstance(obj, pd.DataFrame):
        if return_to_mtype:
            return [obj], "df-list"
        else:
            return [obj]

    if isinstance(obj, np.ndarray):
        if len(obj.shape) == 2:
            obj = np.expand_dims(obj, 0)
            obj = np.swapaxes(obj, 1, 2)
            obj_mtype = "numpy3D"
        elif len(obj.shape) == 1:
            obj = np.expand_dims(obj, (0, 1))
            obj_mtype = "numpy3D"
        else:
            raise ValueError("if obj is np.ndarray, must be of dim 1 or 2")
        if return_to_mtype:
            return obj, obj_mtype
        else:
            return obj

    if _HAS_POLARS and isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        res = obj.with_columns(pl.lit(0).alias("__index__instances"))
        cols = ["__index__instances"] + [
            c for c in obj.columns if c != "__index__instances"
        ]
        res = res.select(cols)
        obj_mtype = "polars_panel"
        if return_to_mtype:
            return res, obj_mtype
        else:
            return res

    raise TypeError(
        "obj must be of a supported Series mtype "
        "(pd.DataFrame, pd.Series, np.ndarray, or polars DataFrame/LazyFrame), "
        f"found {type(obj)}"
    )


def convert_Panel_to_Series(obj, store=None, return_to_mtype=False):
    """Convert single-series panel to a series.

    Removes panel index from the single-series panel to obtain a series.

    Assumes input is conformant with one of the panel mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Panel
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    if obj df-list or pd-multiindex, returns a series of type pd.DataFrame
    if obj was numpy3D, returns a panel mtype np.ndarray
    if obj was polars_panel, returns polars Series container
    """
    if isinstance(obj, list):
        if len(obj) == 1:
            if return_to_mtype:
                return obj[0], "pd.DataFrame"
            else:
                return obj[0]
        else:
            raise ValueError("obj must be of length 1")

    if isinstance(obj, pd.DataFrame):
        obj = obj.copy()
        obj.index = obj.index.droplevel(level=0)
        obj_mtype = "pd.DataFrame"
        if return_to_mtype:
            return obj, obj_mtype
        else:
            return obj

    if isinstance(obj, np.ndarray):
        if obj.ndim != 3 or obj.shape[0] != 1:
            raise ValueError("if obj is np.ndarray, must be of dim 3, with shape[0]=1")
        obj = np.reshape(obj, (obj.shape[1], obj.shape[2]))
        obj = np.swapaxes(obj, 0, 1)
        obj_mtype = "np.ndarray"
        if return_to_mtype:
            return obj, obj_mtype
        else:
            return obj

    if _HAS_POLARS and isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        mi_cols = get_mi_cols(obj)
        if len(mi_cols) == 0:
            raise ValueError("obj has no index columns to identify panel instances")

        instance_col = mi_cols[0]
        # check single-series panel
        if isinstance(obj, pl.DataFrame):
            n_instances = obj[instance_col].n_unique()
        else:
            n_instances = obj.select(pl.col(instance_col).n_unique()).collect().item()

        if n_instances > 1:
            raise ValueError(
                "obj must be a single-series panel, but has multiple instances"
            )

        res = obj.drop(instance_col)
        obj_mtype = "polars_series"
        if return_to_mtype:
            return res, obj_mtype
        else:
            return res

    raise TypeError(
        f"obj must be of a supported Panel mtype (df-list, pd-multiindex, numpy3D, "
        f"or polars_panel), found {type(obj)}"
    )


def convert_Series_to_Hierarchical(obj, store=None, return_to_mtype=False):
    """Convert series to a single-series hierarchical object."""
    if _HAS_POLARS and isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        target_mtype = "polars_hierarchical"
        res = obj.with_columns(
            [
                pl.lit(0).alias("__index__hier0"),
                pl.lit(0).alias("__index__hier1"),
            ]
        )
        cols = ["__index__hier0", "__index__hier1"] + [
            c for c in obj.columns if c not in ("__index__hier0", "__index__hier1")
        ]
        res = res.select(cols)
        if return_to_mtype:
            return res, target_mtype
        else:
            return res

    target_mtype = "pd_multiindex_hier"
    as_scitype = "Series"
    obj_df = convert_to(obj, to_type="pd.DataFrame", as_scitype=as_scitype)
    obj_df = obj_df.copy()
    obj_df["__level1"] = 0
    obj_df["__level2"] = 0
    obj_df = obj_df.set_index(["__level1", "__level2"], append=True)
    obj_df = obj_df.reorder_levels([1, 2, 0])

    if return_to_mtype:
        return obj_df, target_mtype
    else:
        return obj_df


def convert_Hierarchical_to_Series(obj, store=None, return_to_mtype=False):
    """Convert single-series hierarchical object to a series."""
    if _HAS_POLARS and isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        mi_cols = get_mi_cols(obj)
        # remove top hierarchy levels, keep only the time index if present
        if len(mi_cols) >= 2:
            drop_cols = mi_cols[:-1] if len(mi_cols) > 1 else mi_cols
            res = obj.drop(drop_cols)
        else:
            res = obj
        if return_to_mtype:
            return res, "polars_series"
        else:
            return res

    obj_df = convert_to(obj, to_type="pd_multiindex_hier", as_scitype="Hierarchical")
    obj_df = obj_df.copy()
    obj_df.index = obj_df.index.get_level_values(-1)

    if return_to_mtype:
        return obj_df, "pd.DataFrame"
    else:
        return obj_df


def convert_Panel_to_Hierarchical(obj, store=None, return_to_mtype=False):
    """Convert panel to a single-panel hierarchical object."""
    if _HAS_POLARS and isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        res = obj.with_columns(pl.lit(0).alias("__index__hier0"))
        cols = ["__index__hier0"] + [c for c in obj.columns if c != "__index__hier0"]
        res = res.select(cols)
        if return_to_mtype:
            return res, "polars_hierarchical"
        else:
            return res

    obj_df = convert_to(obj, to_type="pd-multiindex", as_scitype="Panel")
    obj_df = obj_df.copy()
    obj_df["__level2"] = 0
    obj_df = obj_df.set_index(["__level2"], append=True)
    obj_df = obj_df.reorder_levels([2, 0, 1])

    if return_to_mtype:
        return obj_df, "pd_multiindex_hier"
    else:
        return obj_df


def convert_Hierarchical_to_Panel(obj, store=None, return_to_mtype=False):
    """Convert single-series hierarchical object to a panel."""
    if _HAS_POLARS and isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        mi_cols = get_mi_cols(obj)
        if len(mi_cols) > 0:
            top_level = mi_cols[0]
            if isinstance(obj, pl.DataFrame):
                n_top = obj[top_level].n_unique()
            else:
                n_top = obj.select(pl.col(top_level).n_unique()).collect().item()
            if n_top > 1:
                raise ValueError(
                    "obj must have a single top-level hierarchy level, found multiple"
                )
            res = obj.drop(top_level)
        else:
            res = obj
        if return_to_mtype:
            return res, "polars_panel"
        else:
            return res

    obj_df = convert_to(obj, to_type="pd_multiindex_hier", as_scitype="Hierarchical")
    obj_df = obj_df.copy()
    obj_df.index = obj_df.index.droplevel(level=0)

    if return_to_mtype:
        return obj_df, "pd-multiindex"
    else:
        return obj_df


def convert_to_scitype(
    obj, to_scitype, from_scitype=None, store=None, return_to_mtype=False
):
    """Convert object to a different scitype."""
    if from_scitype is None:
        from_scitype = scitype(
            obj, candidate_scitypes=["Series", "Panel", "Hierarchical"]
        )

    if to_scitype == from_scitype:
        return obj

    func_name = f"convert_{from_scitype}_to_{to_scitype}"
    func = eval(func_name)

    return func(obj, store=store, return_to_mtype=return_to_mtype)
