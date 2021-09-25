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
from typing import Union

from sktime.datatypes._series._registry import (
    MTYPE_REGISTER_SERIES,
    MTYPE_LIST_SERIES,
)

from sktime.datatypes._panel._registry import (
    MTYPE_REGISTER_PANEL,
    MTYPE_LIST_PANEL,
)

from sktime.base._registry_enum import BaseRegistryEnum
from sktime.datatypes._panel._registry import PanelMtype
from sktime.datatypes._series._registry import SeriesMtype


MTYPE_REGISTER = MTYPE_REGISTER_SERIES + MTYPE_REGISTER_PANEL


__all__ = [
    "MTYPE_REGISTER",
    "MTYPE_LIST_PANEL",
    "MTYPE_LIST_SERIES",
    "SCITYPE_REGISTER",
    "Scitype",
]


class Scitype(BaseRegistryEnum):
    """Enum class for scitypes."""

    series = ("Series", "uni- or multivariate time series")
    panel = ("Panel", "panel of uni- or multivariate time series")


SCITYPE_REGISTER = [tuple(scitype) for scitype in Scitype]


def mtype_to_scitype(mtype: Union[PanelMtype, SeriesMtype, str]) -> str:
    """Infer scitype belonging to mtype.

    Parameters
    ----------
    mtype: str or PanelMtype enum or SeriesMtype enum
        mtype to find scitype of
    Returns
    -------
    scitype: str, unique scitype belonging to mtype
    Raises
    ------
    ValueError, if there are two scitypes with that mtype
        (this should not happen in general)
    ValueError, if there is no scitype with that mtype
    """
    if mtype is not None:
        mtype = str(mtype)
    scitype = [k[1] for k in MTYPE_REGISTER if k[0] == mtype]

    if len(scitype) > 1:
        raise ValueError("multiple scitypes match the mtype, specify scitype")

    if len(scitype) < 1:
        raise ValueError(f"{mtype} is not a supported mtype")

    return scitype[0]
