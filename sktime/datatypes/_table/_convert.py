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

from sktime.datatypes._table._registry import MTYPE_LIST_TABLE

##############################################################
# methods to convert one machine type to another machine type
##############################################################

convert_dict = dict()


def convert_identity(obj, store=None):

    return obj


# assign identity function to type conversion to self
for tp in MTYPE_LIST_TABLE:
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


def convert_s_to_df_as_table(obj: pd.Series, store=None) -> pd.DataFrame:

    if not isinstance(obj, pd.Series):
        raise TypeError("input must be a pd.Series")

    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == 1
    ):
        res = pd.DataFrame(obj, columns=store["columns"])
    else:
        res = pd.DataFrame(obj)

    return res


convert_dict[
    ("pd_Series_Table", "pd_DataFrame_Table", "Table")
] = convert_s_to_df_as_table


def convert_df_to_s_as_table(obj: pd.DataFrame, store=None) -> pd.Series:

    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input is not a pd.DataFrame")

    if len(obj.columns) != 1:
        raise ValueError("input must be univariate pd.DataFrame, with one column")

    if isinstance(store, dict):
        store["columns"] = obj.columns[[0]]

    y = obj[obj.columns[0]]
    y.name = None

    return y


convert_dict[
    ("pd_DataFrame_Table", "pd_Series_Table", "Table")
] = convert_df_to_s_as_table


def convert_list_of_dict_to_df_as_table(obj: list, store=None) -> pd.DataFrame:

    if not isinstance(obj, list):
        raise TypeError("input must be a list of dict")

    if not np.all([isinstance(x, dict) for x in obj]):
        raise TypeError("input must be a list of dict")

    res = pd.DataFrame(obj)

    if (
        isinstance(store, dict)
        and "index" in store.keys()
        and len(store["index"]) == len(res)
    ):
        res.index = store["index"]

    return res


convert_dict[
    ("list_of_dict", "pd_DataFrame_Table", "Table")
] = convert_list_of_dict_to_df_as_table


def convert_df_to_list_of_dict_as_table(obj: pd.DataFrame, store=None) -> list:

    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input is not a pd.DataFrame")

    ret_dict = [obj.loc[i].to_dict() for i in obj.index]

    if isinstance(store, dict):
        store["index"] = obj.index

    return ret_dict


convert_dict[
    ("pd_DataFrame_Table", "list_of_dict", "Table")
] = convert_df_to_list_of_dict_as_table


# obtain other conversions from/to numpyflat via concatenation to DataFrame
def _concat(fun1, fun2):
    def concat_fun(obj, store=None):
        obj1 = fun1(obj, store=store)
        obj2 = fun2(obj1, store=store)
        return obj2

    return concat_fun


def _extend_conversions(mtype, anchor_mtype, convert_dict, mtype_list=None):

    keys = convert_dict.keys()
    scitype = list(keys)[0][2]
    if mtype_list is None:
        mtype_list = [key[0] for key in keys]

    for tp in set(MTYPE_LIST_TABLE).difference([mtype, anchor_mtype]):
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


_extend_conversions("pd_Series_Table", "pd_DataFrame_Table", convert_dict)
_extend_conversions("list_of_dict", "pd_DataFrame_Table", convert_dict)
