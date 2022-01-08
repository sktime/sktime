# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Registry of mtypes and scitypes.

Note for extenders: new mtypes for an existing scitypes
    should be entered in the _registry in the module with name _[scitype].
When adding a new scitype, add it in SCITYPE_REGISTER here.

This module exports the following:

---

SCITYPE_REGISTER - list of tuples

each tuple corresponds to an mtype tag, elements as follows:
    0 : string - name of the scitype as used throughout sktime and in datatypes
    1 : string - plain English description of the scitype

---

MTYPE_REGISTER - list of tuples

each tuple corresponds to an mtype, elements as follows:
    0 : string - name of the mtype as used throughout sktime and in datatypes
    1 : string - name of the scitype the mtype is for, must be in SCITYPE_REGISTER
    2 : string - plain English description of the scitype

---

mtype_to_scitype(mtype: str) - convenience function that returns scitype for an mtype

---
"""

from sktime.datatypes._alignment._registry import (
    MTYPE_LIST_ALIGNMENT,
    MTYPE_REGISTER_ALIGNMENT,
)
from sktime.datatypes._hierarchical._registry import (
    MTYPE_LIST_HIERARCHICAL,
    MTYPE_REGISTER_HIERARCHICAL,
)
from sktime.datatypes._panel._registry import MTYPE_LIST_PANEL, MTYPE_REGISTER_PANEL
from sktime.datatypes._series._registry import MTYPE_LIST_SERIES, MTYPE_REGISTER_SERIES
from sktime.datatypes._table._registry import MTYPE_LIST_TABLE, MTYPE_REGISTER_TABLE

MTYPE_REGISTER = []
MTYPE_REGISTER += MTYPE_REGISTER_SERIES
MTYPE_REGISTER += MTYPE_REGISTER_PANEL
MTYPE_REGISTER += MTYPE_REGISTER_HIERARCHICAL
MTYPE_REGISTER += MTYPE_REGISTER_ALIGNMENT
MTYPE_REGISTER += MTYPE_REGISTER_TABLE


__all__ = [
    "MTYPE_REGISTER",
    "MTYPE_LIST_HIERARCHICAL",
    "MTYPE_LIST_PANEL",
    "MTYPE_LIST_SERIES",
    "MTYPE_LIST_ALIGNMENT",
    "MTYPE_LIST_TABLE",
    "SCITYPE_REGISTER",
]


SCITYPE_REGISTER = [
    ("Series", "uni- or multivariate time series"),
    ("Panel", "panel of uni- or multivariate time series"),
    ("Hierarchical", "hierarchical panel of time series with 3 or more levels"),
    ("Alignment", "series or sequence alignment"),
    ("Table", "data table with primitive column types"),
]


def mtype_to_scitype(mtype: str, return_unique=False):
    """Infer scitype belonging to mtype.

    Parameters
    ----------
    mtype : str, or list of str, or nested list/str object, or None
        mtype(s) to find scitype of

    Returns
    -------
    scitype : str, or list of str, or nested list/str object, or None
        if str, returns scitype belonging to mtype, if mtype is str
        if list, returns this function element-wise applied
        if nested list/str object, replaces mtype str by scitype str
        if None, returns None
    return_unique : bool, default=False
        if True, makes

    Raises
    ------
    TypeError, if input is not of the type specified
    ValueError, if there are two scitypes for the/some mtype string
        (this should not happen in general, it means there is a bug)
    ValueError, if there is no scitype for the/some mtype string
    """
    # handle the "None" case first
    if mtype is None or mtype == "None":
        return None
    # recurse if mtype is a list
    if isinstance(mtype, list):
        scitype_list = [mtype_to_scitype(x) for x in mtype]
        if return_unique:
            scitype_list = list(set(scitype_list))
        return scitype_list

    # checking for type. Checking str is enough, recursion above will do the rest.
    if not isinstance(mtype, str):
        raise TypeError(
            "mtype must be str, or list of str, nested list/str object, or None"
        )

    scitype = [k[1] for k in MTYPE_REGISTER if k[0] == mtype]

    if len(scitype) > 1:
        raise ValueError("multiple scitypes match the mtype, specify scitype")

    if len(scitype) < 1:
        raise ValueError(f"{mtype} is not a supported mtype")

    return scitype[0]
