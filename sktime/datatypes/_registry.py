# -*- coding: utf-8 -*-

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

    if len(scitype > 1):
        raise ValueError("multiple scitypes match the mtype, specify scitype")

    if len(scitype < 1):
        raise ValueError(f"{mtype} is not a supported mtype")

    return scitype[0]
