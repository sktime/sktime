# -*- coding: utf-8 -*-
"""
Machine type converters for scitypes

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""


__author__ = ["fkiraly"]

__all__ = ["convert"]


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
