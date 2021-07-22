# -*- coding: utf-8 -*-
"""Machine type converters for scitypes.

Exports
-------
convert_to(obj, to_type: str, as_scitype: str, store=None)
    converts object "obj" to type "to_type", considerd as "as_scitype"

convert(obj, from_type: str, to_type: str, as_scitype: str, store=None)
    same as convert_to, without automatic identification of "from_type"

mtype(obj, as_scitype: str)
    returns "from_type" of obj, considered as "as_scitype"
---

Function signature of convert

Parameters
----------
obj : object to convert - any type, should comply with mtype spec for as_scitype
from_type : str - the type to convert "obj" to, a valid mtype string
to_type : str - the type to convert "obj" to, a valid mtype string
as_scitype : str - name of scitype the object "obj" is considered as
store : reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

---

Function signature of convert_to

Parameters
----------
obj : object to convert - any type, should comply with mtype spec for as_scitype
to_type : str - the type to convert "obj" to, a valid mtype string
as_scitype : str - name of scitype the object "obj" is considered as
store : reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

"""


__author__ = ["fkiraly"]

__all__ = [
    "convert",
    "convert_to",
    "mtype",
]


import numpy as np
import pandas as pd


##############################################################
# methods to convert one machine type to another machine type
##############################################################

""" key objects in this section:

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

convert_dict = dict()


def convert_identity(obj, store=None):

    return obj


# assign identity function to type conversion to self
for tp in ["pd.Series", "pd.DataFrame", "np.ndarray"]:
    convert_dict[(tp, tp, "Series")] = convert_identity


def convert_UvS_to_MvS_as_Series(obj: pd.Series, store=None) -> pd.DataFrame:

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


convert_dict[("pd.Series", "pd.DataFrame", "Series")] = convert_UvS_to_MvS_as_Series


def convert_MvS_to_UvS_as_Series(obj: pd.DataFrame, store=None) -> pd.Series:

    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input is not a pd.DataFrame")

    if len(obj.columns) != 1:
        raise ValueError("pd.DataFrame must be pd.DataFrame with one column")

    if isinstance(store, dict):
        store["columns"] = obj.columns[[0]]

    return obj[obj.columns[0]]


convert_dict[("pd.Series", "pd.DataFrame", "Series")] = convert_MvS_to_UvS_as_Series


def convert_MvS_to_np_as_Series(obj: pd.DataFrame, store=None) -> np.array:

    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input must be a pd.DataFrame")

    if isinstance(store, dict):
        store["columns"] = obj.columns

    return obj.to_numpy()


convert_dict[("pd.DataFrame", "np.array", "Series")] = convert_MvS_to_np_as_Series


def convert_UvS_to_np_as_Series(obj: pd.Series, store=None) -> np.array:

    if not isinstance(obj, pd.Series):
        raise TypeError("input must be a pd.Series")

    return pd.DataFrame(obj).to_numpy()


convert_dict[("pd.Series", "np.ndarray", "Series")] = convert_UvS_to_np_as_Series


def convert_np_to_MvS_as_Series(obj: np.array, store=None) -> pd.DataFrame:

    if not isinstance(obj, np.array) and len(obj.shape) != 2:
        raise TypeError("input must be a np.array of dim 2")

    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == obj.shape[1]
    ):
        res = pd.DataFrame(obj, columns=store["columns"])
    else:
        res = pd.DataFrame(obj)

    return res


convert_dict[("np.ndarray", "pd.DataFrame", "Series")] = convert_np_to_MvS_as_Series


def convert_np_to_UvS_as_Series(obj: np.array, store=None) -> pd.Series:

    if not isinstance(obj, np.array) and len(obj.shape) < 3:
        raise TypeError("input must be a np.array of dim 1 or 2")

    return pd.Series(obj)


convert_dict[("np.ndarray", "pd.Series", "Series")] = convert_np_to_UvS_as_Series


#########################################################
# methods to infer the machine type subject to a scitype
#########################################################

infer_mtype_dict = dict()


def infer_mtype_Series(obj):

    obj_type = type(obj)

    infer_dict = {
        pd.Series: "pd.Series",
        pd.DataFrame: "pd.DataFrame",
        np.ndarray: "np.ndarray",
    }

    if obj_type not in infer_dict.keys():
        raise TypeError("scitype cannot be inferred")

    return infer_dict[obj_type]


infer_mtype_dict["Series"] = infer_mtype_Series


##################################################
# public functions - mtype inspection, conversion
##################################################


def mtype(obj, as_scitype: str):
    """Infer the mtype of an object considered as a specific scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    as_scitype : str - name of scitype the object "obj" is considered as

    Returns
    -------
    str - the type to convert "obj" to, a valid mtype string
        or None, if obj is None

    Raises
    ------
    TypeError if no type can be identified
    """
    if obj is None:
        return None

    valid_as_scitypes = infer_mtype_dict.keys()

    if as_scitype not in valid_as_scitypes:
        raise TypeError(as_scitype + " is not a valid scitype")

    return infer_mtype_dict[as_scitype](obj=obj)


def convert(obj, from_type: str, to_type: str, as_scitype: str, store=None):
    """Convert objects between different machine representations, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    from_type : str - the type to convert "obj" to, a valid mtype string
    to_type : str - the type to convert "obj" to, a valid mtype string
    as_scitype : str - name of scitype the object "obj" is considered as
    store : reference of storage for lossy conversions, default=None (no store)

    Returns
    -------
    converted_obj : to_type - object obj converted to to_type
                    if obj was None, returns None

    Raises
    ------
    KeyError if conversion is not implemented
    """
    if obj is None:
        return None

    key = (from_type, to_type, as_scitype)

    if key not in convert_dict.keys():
        raise TypeError(
            "no conversion defined from type " + str(from_type) + " to " + str(to_type)
        )

    converted_obj = convert_dict[key](obj, store=store)

    return converted_obj


# conversion based on queriable type to specified target
def convert_to(obj, to_type: str, as_scitype: str, store=None):
    """Convert object to a different machine representation, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    to_type : str - the type to convert "obj" to, a valid mtype string
              or list - admissible types for conversion to
    as_scitype : str - name of scitype the object "obj" is considered as
    store : reference of storage for lossy conversions, default=None (no store)

    Returns
    -------
    converted_obj : to_type - object obj converted to to_type, if to_type is str
                     if to_type is list, converted to to_type[0],
                        unless from_type in to_type, in this case converted_obj=obj
                    if obj was None, returns None

    Raises
    ------
    TypeError if machine type of input "obj" is not recognized
    KeyError if conversion is not implemented
    """
    if obj is None:
        return None

    from_type = mtype(obj=obj, as_scitype=as_scitype)

    # if to_type is a list:
    if isinstance(to_type, list):
        # no conversion of from_type is in the list
        if from_type in to_type:
            to_type = from_type
        # otherwise convert to first element
        else:
            to_type = to_type[0]

    converted_obj = convert(
        obj=obj,
        from_type=from_type,
        to_type=to_type,
        as_scitype=as_scitype,
        store=store,
    )

    return converted_obj
