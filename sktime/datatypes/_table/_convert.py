# -*- coding: utf-8 -*-
"""Machine type converters for Table scitype.

Exports conversion and mtype dictionary for Table scitype:

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

convert_dict = dict()


def convert_identity(obj, store=None):

    return obj


# assign identity function to type conversion to self
for tp in ["numpy1D", "numpy2D", "pd_DataFrame_Table"]:
    convert_dict[(tp, tp, "Table")] = convert_identity


def convert_1D_to_2D_numpy_as_Table(obj: np.ndarray, store=None) -> np.ndarray:

    if not isinstance(obj, np.ndarray):
        raise TypeError("input must be a np.ndarray")

    if len(obj.shape) == 1:
        res = np.reshape(obj, (-1, 1))
    else:
        raise TypeError("input must be 1D np.ndarray")

    return res


convert_dict[("numpy1D", "numpy2D", "Table")] = convert_1D_to_2D_numpy_as_Table


def convert_2D_to_1D_numpy_as_Table(obj: np.ndarray, store=None) -> np.ndarray:

    if not isinstance(obj, np.ndarray):
        raise TypeError("input must be a np.ndarray")

    if len(obj.shape) == 2:
        res = obj.flatten()
    else:
        raise TypeError("input must be 2D np.ndarray")

    return res


convert_dict[("numpy2D", "numpy1D", "Table")] = convert_2D_to_1D_numpy_as_Table


def convert_df_to_2Dnp_as_Table(obj: pd.DataFrame, store=None) -> np.ndarray:

    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input must be a pd.DataFrame")

    if isinstance(store, dict):
        store["columns"] = obj.columns

    return obj.to_numpy()


convert_dict[("pd_DataFrame_Table", "numpy2D", "Table")] = convert_df_to_2Dnp_as_Table


def convert_df_to_1Dnp_as_Table(obj: pd.DataFrame, store=None) -> np.ndarray:

    return convert_df_to_2Dnp_as_Table(obj=obj, store=store).flatten()


convert_dict[("pd_DataFrame_Table", "numpy1D", "Table")] = convert_df_to_1Dnp_as_Table


def convert_2Dnp_to_df_as_Table(obj: np.ndarray, store=None) -> pd.DataFrame:

    if not isinstance(obj, np.ndarray) and len(obj.shape) != 2:
        raise TypeError("input must be a 2D np.ndarray")

    if len(obj.shape) == 1:
        obj = np.reshape(obj, (-1, 1))

    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == obj.shape[1]
    ):
        res = pd.DataFrame(obj, columns=store["columns"])
    else:
        res = pd.DataFrame(obj)

    return res


convert_dict[("numpy2D", "pd_DataFrame_Table", "Table")] = convert_2Dnp_to_df_as_Table


def convert_1Dnp_to_df_as_Table(obj: np.ndarray, store=None) -> pd.DataFrame:

    if not isinstance(obj, np.ndarray) and len(obj.shape) != 1:
        raise TypeError("input must be a 1D np.ndarray")

    obj = np.reshape(obj, (-1, 1))

    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == obj.shape[1]
    ):
        res = pd.DataFrame(obj, columns=store["columns"])
    else:
        res = pd.DataFrame(obj)

    return res


convert_dict[("numpy1D", "pd_DataFrame_Table", "Table")] = convert_1Dnp_to_df_as_Table
