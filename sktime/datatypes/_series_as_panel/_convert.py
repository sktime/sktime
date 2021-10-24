# -*- coding: utf-8 -*-
"""Machine type converters for Series to Panel.

Exports conversion dictionary for conversions between Series and Panel:

convert_dict: dict indexed by triples of str
  1st element = convert from - str
  2nd element = convert to - str
  3rd element = considered as this scitype - str
elements are conversion functions of machine type (1st) -> 2nd

Function signature of all elements
convert_dict[(from_type, to_type, as_scitype)]

Parameters
----------
obj : from_type - object to convert
store : dictionary - reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

Raises
------
ValueError and TypeError, if requested conversion is not possible
                            (depending on conversion logic)
"""

__author__ = ["fkiraly"]

__all__ = ["convert_Series_to_Panel", "convert_Panel_to_Series"]

import numpy as np
import pandas as pd


def convert_Series_to_Panel(obj, store=None):
    """Convert series to a single-series panel.

    Assumes input is conformant with three scitypes.
    This method does not perform full mtype checks, use mtype or check_is for checks.

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
        obj = np.expand_dims(obj, 2)

    return obj


def convert_Panel_to_Series(obj, store=None):
    """Convert single-series panel to a series.

    Assumes input is one of three main panel scitypes.
    This method does not perform full mtype checks, use mtype or check_is for checks.

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
            raise ValueError
        obj = np.reshape(obj, (obj.shape[0], obj.shape[1]))

    return obj
