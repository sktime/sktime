# -*- coding: utf-8 -*-
"""Machine type converters for Series to Panel.

Exports conversion functions for conversions between Series and Panel:

convert_Series_to_Panel(obj, store=None)
    converts obj of Series mtype to "adjacent" Panel mtype (e.g., numpy to numpy)
convert_Panel_to_Series(obj, store=None)
    converts obj of Panel mtype to "adjacent" Series mtype (e.g., numpy to numpy)
"""

__author__ = ["fkiraly"]

__all__ = ["convert_Series_to_Panel", "convert_Panel_to_Series"]

import numpy as np
import pandas as pd


def convert_Series_to_Panel(obj, store=None):
    """Convert series to a single-series panel.

    Assumes input is conformant with one of the three Series mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Series, of mtype pd.DataFrame, pd.Series, or np.ndarray.

    Returns
    -------
    if obj was pd.Series or pd.DataFrame, returns a panel of mtype df-list
        this is done by possibly converting to pd.DataFrame, and adding a list nesting
    if obj was np.ndarray, returns a panel of mtype numpy3D, by adding one axis at end
    """
    if isinstance(obj, pd.Series):
        obj = pd.DataFrame(obj)

    if isinstance(obj, pd.DataFrame):
        return [obj]

    if isinstance(obj, np.ndarray):
        if len(obj.shape) == 2:
            obj = np.expand_dims(obj, 2)
        elif len(obj.shape) == 1:
            obj = np.expand_dims(obj, (1, 2))
        else:
            raise ValueError("if obj is np.ndarray, must be of dim 1 or 2")

    return obj


def convert_Panel_to_Series(obj, store=None):
    """Convert single-series panel to a series.

    Assumes input is conformant with one of three main panel mtypes.
    This method does not perform full mtype checks, use mtype or check_is_mtype for
    checks.

    Parameters
    ----------
    obj: an object of scitype Panel, of mtype pd-multiindex, numpy3d, or df-list.

    Returns
    -------
    if obj df-list or pd-multiindex, returns a series of type pd.DataFrame
    if obj was numpy3D, returns a panel mtype np.ndarray
    """
    if isinstance(obj, list):
        if len(obj) == 1:
            return obj[0]
        else:
            raise ValueError("obj must be of length 1")

    if isinstance(obj, pd.DataFrame):
        obj.index = obj.index.droplevel(level=0)

    if isinstance(obj, np.ndarray):
        shape = obj.shape
        if not len(shape == 3) or not shape[2] == 1:
            raise ValueError("if obj is np.ndarray, must be of dim 3, with shape[2]=1")
        obj = np.reshape(obj, (obj.shape[0], obj.shape[1]))

    return obj
