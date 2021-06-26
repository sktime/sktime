# -*- coding: utf-8 -*-
"""
Machine type converters for scitypes

Exports
-------
convert : dictionary of functions, with signature as below
        indexed by triples of types
         1st element = convert from - type
         2nd element = convert to - type
         3rd element = considered as this scitype - string
        elements are conversion functions of machine type (1st) -> 2nd

convert_to : function/dispatch version of convert, with 1st elt inferred


-----------------------------------------

Function signature of
convert[(from_type, to_type, as_scitype)]

Parameters
----------
what : from_type - object to convert
store : dictionary - reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_what : to_type - object what converted to to_type

Raises
------
ValueError and TypeError, if requested conversion is not possible
                            (depending on conversion logic)

-----------------------------------------

Function signature of convert_to

Parameters
----------
what : object to convert (any type)
to_type : type, the type to convert "what" to
as_scitype : str - name of scitype the object "what" is considered as
store : reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_what : to_type - object what converted to to_type

Raises
------
TypeError if no suitable key in convert exists
    i.e., (type(what), to_type, as_scitype)


copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""


__author__ = ["fkiraly"]


import numpy as np
import pandas as pd


# dictionary indexed by triples of types
#  1st element = convert from - type
#  2nd element = convert to - type
#  3rd element = considered as this scitype - string
# elements are conversion functions of machine type (1st) -> 2nd

convert = dict()


def convert_identity(what, store=None):

    return what


# assign identity function to type conversion to self
for tp in [pd.Series, pd.DataFrame, np.array]:
    convert[(tp, tp, "Series")] = convert_identity


def convert_UvS_to_MvS_as_Series(what: pd.Series, store=None) -> pd.DataFrame:

    if not isinstance(what, pd.Series):
        raise TypeError("input must be a pd.Series")

    if isinstance(store, dict) and "cols" in store.keys and len(store["cols"]) == 1:
        res = pd.DataFrame(what, columns=store["cols"])
    else:
        res = pd.DataFrame(what)

    return res


convert[(pd.Series, pd.DataFrame, "Series")] = convert_UvS_to_MvS_as_Series


def convert_MvS_to_UvS_as_Series(what: pd.DataFrame, store=None) -> pd.Series:

    if not isinstance(what, pd.DataFrame):
        raise TypeError("input is not a pd.DataFrame")

    if len(what.columns) != 1:
        raise ValueError("pd.DataFrame must be pd.DataFrame with one column")

    if isinstance(store, dict):
        store["cols"] = what.columns[[0]]

    return what[what.columns[0]]


convert[(pd.Series, pd.DataFrame, "Series")] = convert_MvS_to_UvS_as_Series


def convert_MvS_to_np_as_Series(what: pd.DataFrame, store=None) -> np.array:

    if not isinstance(what, pd.DataFrame):
        raise TypeError("input must be a pd.DataFrame")

    if isinstance(store, dict):
        store["cols"] = what.columns

    return what.to_numpy()


convert[(pd.DataFrame, np.array, "Series")] = convert_MvS_to_np_as_Series


def convert_UvS_to_np_as_Series(what: pd.Series, store=None) -> np.array:

    if not isinstance(what, pd.Series):
        raise TypeError("input must be a pd.Series")

    return pd.DataFrame(what).to_numpy()


convert[(pd.Series, np.array, "Series")] = convert_UvS_to_np_as_Series


def convert_np_to_MvS_as_Series(what: np.array, store=None) -> pd.DataFrame:

    if not isinstance(what, np.array) and len(what.shape) != 2:
        raise TypeError("input must be a np.array of dim 2")

    if (
        isinstance(store, dict)
        and "cols" in store.keys
        and len(store["cols"]) == what.shape[1]
    ):
        res = pd.DataFrame(what, columns=store["cols"])
    else:
        res = pd.DataFrame(what)

    return res


convert[(np.array, pd.DataFrame, "Series")] = convert_np_to_MvS_as_Series


def convert_np_to_UvS_as_Series(what: np.array, store=None) -> pd.Series:

    if not isinstance(what, np.array) and len(what.shape) < 3:
        raise TypeError("input must be a np.array of dim 1 or 2")

    return pd.Series(what)


convert[(np.array, pd.Series, "Series")] = convert_np_to_UvS_as_Series


# conversion based on queriable type to specified target
def convert_to(what, to_type: type, as_scitype: str, store=None):

    from_type = type(what)

    key = (from_type, to_type, as_scitype)

    ckys = list(convert.keys())

    if key not in ckys:
        raise TypeError(
            "no conversion defined from type " + str(from_type) + " to " + str(to_type)
        )

    return convert[key](what, store=store)
