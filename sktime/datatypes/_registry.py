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

from sktime.datatypes._series._registry import (
    MTYPE_REGISTER_SERIES,
    MTYPE_LIST_SERIES,
)

from sktime.datatypes._panel._registry import (
    MTYPE_REGISTER_PANEL,
    MTYPE_LIST_PANEL,
)

MTYPE_REGISTER = MTYPE_REGISTER_SERIES + MTYPE_REGISTER_PANEL


__all__ = [
    "MTYPE_REGISTER",
    "MTYPE_LIST_PANEL",
    "MTYPE_LIST_SERIES",
    "SCITYPE_REGISTER",
]


SCITYPE_REGISTER = [
    ("Series", "uni- or multivariate time series"),
    ("Panel", "panel of uni- or multivariate time series"),
]


def mtype_to_scitype(mtype: str):
    """Infer scitype belonging to mtype.

    Parameters
    ----------
    mtype: str, mtype to find scitype of

    Returns
    -------
    scitype: str, unique scitype belonging to mtype

    Raises
    ------
    ValueError, if there are two scitypes with that mtype
        (this should not happen in general)
    ValueError, if there is no scitype with that mtype
    """
    scitype = [k[1] for k in MTYPE_REGISTER if k[0] == mtype]

    if len(scitype) > 1:
        raise ValueError("multiple scitypes match the mtype, specify scitype")

    if len(scitype) < 1:
        raise ValueError(f"{mtype} is not a supported mtype")

    return scitype[0]
