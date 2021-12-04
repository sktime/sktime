# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
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

---

Function signature of mtype

Parameters
----------
obj : object to convert - any type, should comply with mtype spec for as_scitype
as_scitype : str - name of scitype the object "obj" is considered as

Returns
-------
str - the type to convert "obj" to, a valid mtype string
    or None, if obj is None
"""

__author__ = ["fkiraly"]

__all__ = [
    "convert",
    "convert_to",
]

import numpy as np
import pandas as pd

from sktime.datatypes._series import convert_dict_Series

from sktime.datatypes._panel import convert_dict_Panel

from sktime.datatypes._check import mtype as infer_mtype
from sktime.datatypes._registry import mtype_to_scitype

# pool convert_dict-s and infer_mtype_dict-s
convert_dict = dict()
convert_dict.update(convert_dict_Series)
convert_dict.update(convert_dict_Panel)


def convert(obj, from_type: str, to_type: str, as_scitype: str = None, store=None):
    """Convert objects between different machine representations, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    from_type : str - the type to convert "obj" to, a valid mtype string
    to_type : str - the type to convert "obj" to, a valid mtype string
    as_scitype : str, optional - name of scitype the object "obj" is considered as
        default = inferred from from_type
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

    if not isinstance(to_type, str):
        raise TypeError("to_type must be a str")
    if not isinstance(from_type, str):
        raise TypeError("from_type must be a str")
    if as_scitype is None:
        as_scitype = mtype_to_scitype(to_type)
    elif not isinstance(as_scitype, str):
        raise TypeError("as_scitype must be str or None")

    key = (from_type, to_type, as_scitype)

    if key not in convert_dict.keys():
        raise NotImplementedError(
            "no conversion defined from type " + str(from_type) + " to " + str(to_type)
        )

    converted_obj = convert_dict[key](obj, store=store)

    return converted_obj


# conversion based on queriable type to specified target
def convert_to(obj, to_type: str, as_scitype: str = None, store=None):
    """Convert object to a different machine representation, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    to_type : str - the type to convert "obj" to, a valid mtype string
              or list - admissible types for conversion to
    as_scitype : str, optional - name of scitype the object "obj" is considered as
        default = inferred from mtype of obj, which is in turn inferred internally
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

    if isinstance(to_type, list):
        if not np.all(isinstance(x, str) for x in to_type):
            raise TypeError("to_type must be a str or list of str")
    elif not isinstance(to_type, str):
        raise TypeError("to_type must be a str or list of str")

    if as_scitype is None:
        if isinstance(to_type, str):
            as_scitype = mtype_to_scitype(to_type)
        else:
            as_scitype = mtype_to_scitype(to_type[0])
    elif not isinstance(as_scitype, str):
        raise TypeError("as_scitype must be a str or None")

    from_type = infer_mtype(obj=obj, as_scitype=as_scitype)

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


def _conversions_defined(scitype: str):
    """Return an indicator matrix which conversions are defined for scitype.

    Parameters
    ----------
    scitype: str - name of scitype for which conversions are queried

    Returns
    -------
    conv_df: pd.DataFrame, columns and index is list of mtypes for scitype
            entry of row i, col j is 1 if conversion from i to j is defined,
                                     0 if conversion from i to j is not defined
    """
    pairs = [(x[0], x[1]) for x in list(convert_dict.keys()) if x[2] == scitype]
    cols0 = set([x[0] for x in list(convert_dict.keys()) if x[2] == scitype])
    cols1 = set([x[1] for x in list(convert_dict.keys()) if x[2] == scitype])
    cols = sorted(list(cols0.union(cols1)))

    mat = np.zeros((len(cols), len(cols)), dtype=int)
    nkeys = len(cols)
    for i in range(nkeys):
        for j in range(nkeys):
            if (cols[i], cols[j]) in pairs:
                mat[i, j] = 1

    conv_df = pd.DataFrame(mat, index=cols, columns=cols)

    return conv_df
