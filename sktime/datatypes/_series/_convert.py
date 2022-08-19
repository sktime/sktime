# -*- coding: utf-8 -*-
"""Machine type converters for Series scitype.

Exports conversion and mtype dictionary for Series scitype:

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

__all__ = ["convert_dict"]

import numpy as np
import pandas as pd

##############################################################
# methods to convert one machine type to another machine type
##############################################################
from sktime.datatypes._registry import MTYPE_LIST_SERIES
from sktime.utils.validation._dependencies import _check_soft_dependencies

convert_dict = dict()


def convert_identity(obj, store=None):

    return obj


# assign identity function to type conversion to self
for tp in ["pd.Series", "pd.DataFrame", "np.ndarray", "xr.DataArray"]:
    convert_dict[(tp, tp, "Series")] = convert_identity


def convert_UvS_to_MvS_as_Series(obj: pd.Series, store=None) -> pd.DataFrame:

    if not isinstance(obj, pd.Series):
        raise TypeError("input must be a pd.Series")

    res = pd.DataFrame(obj)

    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == 1
    ):
        res.columns = store["columns"]

    return res


convert_dict[("pd.Series", "pd.DataFrame", "Series")] = convert_UvS_to_MvS_as_Series


def convert_MvS_to_UvS_as_Series(obj: pd.DataFrame, store=None) -> pd.Series:

    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input is not a pd.DataFrame")

    if len(obj.columns) != 1:
        raise ValueError("input must be univariate pd.DataFrame, with one column")

    if isinstance(store, dict):
        store["columns"] = obj.columns[[0]]

    y = obj[obj.columns[0]]
    y.name = None

    return y


convert_dict[("pd.DataFrame", "pd.Series", "Series")] = convert_MvS_to_UvS_as_Series


def convert_MvS_to_np_as_Series(obj: pd.DataFrame, store=None) -> np.ndarray:

    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input must be a pd.DataFrame")

    if isinstance(store, dict):
        store["columns"] = obj.columns
        store["index"] = obj.index

    return obj.to_numpy(dtype="float")


convert_dict[("pd.DataFrame", "np.ndarray", "Series")] = convert_MvS_to_np_as_Series


def convert_UvS_to_np_as_Series(obj: pd.Series, store=None) -> np.ndarray:

    if not isinstance(obj, pd.Series):
        raise TypeError("input must be a pd.Series")

    if isinstance(store, dict):
        store["index"] = obj.index

    return pd.DataFrame(obj).to_numpy(dtype="float")


convert_dict[("pd.Series", "np.ndarray", "Series")] = convert_UvS_to_np_as_Series


def convert_np_to_MvS_as_Series(obj: np.ndarray, store=None) -> pd.DataFrame:

    if not isinstance(obj, np.ndarray) and len(obj.shape) > 2:
        raise TypeError("input must be a np.ndarray of dim 1 or 2")

    if len(obj.shape) == 1:
        obj = np.reshape(obj, (-1, 1))

    res = pd.DataFrame(obj)

    # add column names or index from store if stored and length fits
    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == obj.shape[1]
    ):
        res.columns = store["columns"]
    if (
        isinstance(store, dict)
        and "index" in store.keys()
        and len(store["index"]) == obj.shape[0]
    ):
        res.index = store["index"]

    return res


convert_dict[("np.ndarray", "pd.DataFrame", "Series")] = convert_np_to_MvS_as_Series


def convert_np_to_UvS_as_Series(obj: np.ndarray, store=None) -> pd.Series:

    if not isinstance(obj, np.ndarray) or obj.ndim > 2:
        raise TypeError("input must be a one-column np.ndarray of dim 1 or 2")

    if obj.ndim == 2 and obj.shape[1] != 1:
        raise TypeError("input must be a one-column np.ndarray of dim 1 or 2")

    res = pd.Series(obj.flatten())

    # add index from store if stored and length fits
    if (
        isinstance(store, dict)
        and "index" in store.keys()
        and len(store["index"]) == obj.shape[0]
    ):
        res.index = store["index"]

    return res


convert_dict[("np.ndarray", "pd.Series", "Series")] = convert_np_to_UvS_as_Series


# obtain other conversions from/to numpyflat via concatenation to numpy3D
def _concat(fun1, fun2):
    def concat_fun(obj, store=None):
        obj1 = fun1(obj, store=store)
        obj2 = fun2(obj1, store=store)
        return obj2

    return concat_fun


def _extend_conversions(mtype, anchor_mtype, convert_dict):
    keys = convert_dict.keys()
    scitype = list(keys)[0][2]

    for tp in set(MTYPE_LIST_SERIES).difference([mtype, anchor_mtype]):
        if (anchor_mtype, tp, scitype) in convert_dict.keys():
            convert_dict[(mtype, tp, scitype)] = _concat(
                convert_dict[(mtype, anchor_mtype, scitype)],
                convert_dict[(anchor_mtype, tp, scitype)],
            )
        if (tp, anchor_mtype, scitype) in convert_dict.keys():
            convert_dict[(tp, mtype, scitype)] = _concat(
                convert_dict[(tp, anchor_mtype, scitype)],
                convert_dict[(anchor_mtype, mtype, scitype)],
            )


if _check_soft_dependencies("xarray", severity="none"):
    import xarray as xr

    def convert_xrdataarray_to_Mvs_as_Series(
        obj: xr.DataArray, store=None
    ) -> pd.DataFrame:
        if not isinstance(obj, xr.DataArray):
            raise TypeError("input must be a xr.DataArray")

        if isinstance(store, dict):
            store["coords"] = list(obj.coords.keys())

        index = obj.indexes[obj.dims[0]]
        columns = obj.indexes[obj.dims[1]] if len(obj.dims) == 2 else None
        return pd.DataFrame(obj.values, index=index, columns=columns)

    convert_dict[
        ("xr.DataArray", "pd.DataFrame", "Series")
    ] = convert_xrdataarray_to_Mvs_as_Series

    def convert_Mvs_to_xrdatarray_as_Series(
        obj: pd.DataFrame, store=None
    ) -> xr.DataArray:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("input must be a xr.DataArray")

        result = xr.DataArray(obj.values, coords=[obj.index, obj.columns])
        if isinstance(store, dict) and "coords" in store:
            result = result.rename(
                dict(zip(list(result.coords.keys()), store["coords"]))
            )
        return result

    convert_dict[
        ("pd.DataFrame", "xr.DataArray", "Series")
    ] = convert_Mvs_to_xrdatarray_as_Series

    _extend_conversions("xr.DataArray", "pd.DataFrame", convert_dict)
