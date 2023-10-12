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


def convert_Series_to_Panel(obj, store=None, return_to_mtype=False):
    """Convert series to a single-series panel.

    Adds a dummy dimension to the series.
    For pd.Series or DataFrame, this results in a list of DataFram (dim added is list).
    For numpy array, this results in a third dimension being added.

    Assumes input is conformant with one of the three Series mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Series, of mtype pd.DataFrame, pd.Series, or np.ndarray.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    if obj was pd.Series or pd.DataFrame, returns a panel of mtype df-list
        this is done by possibly converting to pd.DataFrame, and adding a list nesting
    if obj was np.ndarray, returns a panel of mtype numpy3D, by adding one axis at end
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
            # from numpy2D to numpy3D
            # numpy2D = (time, variables)
            # numpy3D = (instances, variables, time)
            obj = np.expand_dims(obj, 0)
            obj = np.swapaxes(obj, 1, 2)
            obj_mtype = "numpy3D"
        elif len(obj.shape) == 1:
            # from numpy1D to numpy3D
            # numpy1D = (time)
            # numpy3D = (instances, variables, time)
            obj = np.expand_dims(obj, (0, 1))
            obj_mtype = "numpy3D"
        else:
            raise ValueError("if obj is np.ndarray, must be of dim 1 or 2")

    if return_to_mtype:
        return obj, obj_mtype
    else:
        return obj


def convert_Panel_to_Series(obj, store=None, return_to_mtype=False):
    """Convert single-series panel to a series.

    Removes panel index from the single-series panel to obtain a series.

    Assumes input is conformant with one of three main panel mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Panel, of mtype pd-multiindex, numpy3d, or df-list.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    if obj df-list or pd-multiindex, returns a series of type pd.DataFrame
    if obj was numpy3D, returns a panel mtype np.ndarray
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
        obj.index = obj.index.droplevel(level=0)
        obj_mtype = "pd.DataFrame"

    if isinstance(obj, np.ndarray):
        if obj.ndim != 3 or obj.shape[0] != 1:
            raise ValueError("if obj is np.ndarray, must be of dim 3, with shape[0]=1")
        # from numpy3D to numpy2D
        # numpy2D = (time, variables)
        # numpy3D = (instances, variables, time)
        obj = np.reshape(obj, (obj.shape[1], obj.shape[2]))
        obj = np.swapaxes(obj, 0, 1)
        obj_mtype = "np.ndarray"

    if return_to_mtype:
        return obj, obj_mtype
    else:
        return obj


def convert_Series_to_Hierarchical(obj, store=None, return_to_mtype=False):
    """Convert series to a single-series hierarchical object.

    Adds two dimensions to the series to obtain a 3-level MultiIndex, 2 levels added.

    Assumes input is conformant with one of the three Series mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Series, of mtype pd.DataFrame, pd.Series, or np.ndarray.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    returns a data container of mtype pd_multiindex_hier
    """
    obj_df = convert_to(obj, to_type="pd.DataFrame", as_scitype="Series")
    obj_df = obj_df.copy()
    obj_df["__level1"] = 0
    obj_df["__level2"] = 0
    obj_df = obj_df.set_index(["__level1", "__level2"], append=True)
    obj_df = obj_df.reorder_levels([1, 2, 0])

    if return_to_mtype:
        return obj_df, "pd_multiindex_hier"
    else:
        return obj_df


def convert_Hierarchical_to_Series(obj, store=None, return_to_mtype=False):
    """Convert single-series hierarchical object to a series.

    Removes two dimensions to obtain a series, by removing 2 levels from MultiIndex.

    Assumes input is conformant with Hierarchical mtype.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Hierarchical.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    returns a data container of mtype pd.DataFrame, of scitype Series
    """
    obj_df = convert_to(obj, to_type="pd_multiindex_hier", as_scitype="Hierarchical")
    obj_df = obj_df.copy()
    obj_df.index = obj_df.index.get_level_values(-1)

    if return_to_mtype:
        return obj_df, "pd.DataFrame"
    else:
        return obj_df


def convert_Panel_to_Hierarchical(obj, store=None, return_to_mtype=False):
    """Convert panel to a single-panel hierarchical object.

    Adds a dimensions to the panel to obtain a 3-level MultiIndex, 1 level is added.

    Assumes input is conformant with one of the Panel mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Panel.
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    returns a data container of mtype pd_multiindex_hier
    """
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
    """Convert single-series hierarchical object to a series.

    Removes one dimensions to obtain a panel, by removing 1 level from MultiIndex.

    Assumes input is conformant with Hierarchical mtype.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Hierarchical
    store: dict, optional
        converter store for back-conversion
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    returns a data container of mtype pd-multiindex, of scitype Panel
    """
    obj_df = convert_to(obj, to_type="pd_multiindex_hier", as_scitype="Hierarchical")
    obj_df = obj_df.copy()
    obj_df.index = obj_df.index.get_level_values([-2, -1])

    if return_to_mtype:
        return obj_df, "pd-multiindex"
    else:
        return obj_df


def convert_to_scitype(
    obj,
    to_scitype,
    from_scitype=None,
    store=None,
    return_to_mtype=False,
):
    """Convert single-series or single-panel between mtypes.

    Assumes input is conformant with one of the mtypes
        for one of the scitypes Series, Panel, Hierarchical.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj : an object of scitype Series, Panel, or Hierarchical.
    to_scitype : str, scitype that obj should be converted to
    from_scitype : str, optional. Default = inferred from obj
        scitype that obj is of, and being converted from
        if avoided, function will skip type inference from obj
    store : dict, optional. Converter store for back-conversion.
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    obj of scitype to_scitype
        if converted to or from Hierarchical, the mtype will always be one of
            pd.DataFrame (Series), pd-multiindex (Panel), or pd_multiindex_hier
        if converted to or from Panel, mtype will attempt to keep python type
            e.g., np.ndarray (Series) converted to numpy3D (Panel) or back
            if not possible, will be one of the mtypes with pd.DataFrame python type
    """
    if from_scitype is None:
        from_scitype = scitype(
            obj, candidate_scitypes=["Series", "Panel", "Hierarchical"]
        )

    if to_scitype == from_scitype:
        return obj

    func_name = f"convert_{from_scitype}_to_{to_scitype}"
    func = eval(func_name)

    return func(obj, store=store, return_to_mtype=return_to_mtype)
