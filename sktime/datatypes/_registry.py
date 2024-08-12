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

MTYPE_SOFT_DEPS - dict with str keys and values

keys are mtypes with soft dependencies, values are str or list of str
strings in values are names of soft dependency packages required for the mtype

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
    MTYPE_SOFT_DEPS_HIERARCHICAL,
)
from sktime.datatypes._panel._registry import (
    MTYPE_LIST_PANEL,
    MTYPE_REGISTER_PANEL,
    MTYPE_SOFT_DEPS_PANEL,
)
from sktime.datatypes._proba._registry import MTYPE_LIST_PROBA, MTYPE_REGISTER_PROBA
from sktime.datatypes._series._registry import (
    MTYPE_LIST_SERIES,
    MTYPE_REGISTER_SERIES,
    MTYPE_SOFT_DEPS_SERIES,
)
from sktime.datatypes._table._registry import MTYPE_LIST_TABLE, MTYPE_REGISTER_TABLE

MTYPE_REGISTER = []
MTYPE_REGISTER += MTYPE_REGISTER_SERIES
MTYPE_REGISTER += MTYPE_REGISTER_PANEL
MTYPE_REGISTER += MTYPE_REGISTER_HIERARCHICAL
MTYPE_REGISTER += MTYPE_REGISTER_ALIGNMENT
MTYPE_REGISTER += MTYPE_REGISTER_TABLE
MTYPE_REGISTER += MTYPE_REGISTER_PROBA

MTYPE_SOFT_DEPS = {}
MTYPE_SOFT_DEPS.update(MTYPE_SOFT_DEPS_SERIES)
MTYPE_SOFT_DEPS.update(MTYPE_SOFT_DEPS_PANEL)
MTYPE_SOFT_DEPS.update(MTYPE_SOFT_DEPS_HIERARCHICAL)


# mtypes to exclude in checking since they are ambiguous and rare
AMBIGUOUS_MTYPES = ["numpyflat", "alignment_loc"]

# all time series mtypes excluding ambiguous ones
ALL_TIME_SERIES_MTYPES = (
    list(MTYPE_LIST_PANEL) + list(MTYPE_LIST_SERIES) + list(MTYPE_LIST_HIERARCHICAL)
)
ALL_TIME_SERIES_MTYPES = list(set(ALL_TIME_SERIES_MTYPES).difference(AMBIGUOUS_MTYPES))


__all__ = [
    "MTYPE_REGISTER",
    "MTYPE_LIST_HIERARCHICAL",
    "MTYPE_LIST_PANEL",
    "MTYPE_LIST_SERIES",
    "MTYPE_LIST_ALIGNMENT",
    "MTYPE_LIST_TABLE",
    "MTYPE_LIST_PROBA",
    "MTYPE_SOFT_DEPS",
    "SCITYPE_REGISTER",
]


SCITYPE_REGISTER = [
    ("Series", "uni- or multivariate time series"),
    ("Panel", "panel of uni- or multivariate time series"),
    ("Hierarchical", "hierarchical panel of time series with 3 or more levels"),
    ("Alignment", "series or sequence alignment"),
    ("Table", "data table with primitive column types"),
    ("Proba", "probability distribution or distribution statistics, return types"),
]

SCITYPE_LIST = [x[0] for x in SCITYPE_REGISTER]


def mtype_to_scitype(mtype: str, return_unique=False, coerce_to_list=False):
    """Infer scitype belonging to mtype.

    Parameters
    ----------
    mtype : str, or list of str, or nested list/str object, or None
        mtype(s) to find scitype of, a valid mtype string
        valid mtype strings, with explanation, are in datatypes.MTYPE_REGISTER

    Returns
    -------
    scitype : str, or list of str, or nested list/str object, or None
        if str, returns scitype belonging to mtype, if mtype is str
        if list, returns this function element-wise applied
        if nested list/str object, replaces mtype str by scitype str
        if None, returns None
    return_unique : bool, default=False
        if True, makes return unique
    coerce_to_list : bool, default=False
        if True, coerces rerturn to list, even if one-element

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

    if coerce_to_list:
        return [scitype[0]]
    else:
        return scitype[0]


def scitype_to_mtype(scitype: str, softdeps: str = "exclude"):
    """Return list of all mtypes belonging to scitype.

    Parameters
    ----------
    scitype : str, or list of str
        scitype(s) to find mtypes for, a valid scitype string
        valid scitype strings, with explanation, are in datatypes.SCITYPE_REGISTER
    softdeps : str, optional, default = "exclude"
        whether to return mtypes that require soft dependencies
        "exclude" = only mtypes that do not require soft dependencies are returned
        "present" = only mtypes with soft deps satisfied by the environment are returned
        "all" = all mtypes, irrespective of soft deps satisfied or required, returned
        any other value defaults to "all"

    Returns
    -------
    mtypes : list of str
        all mtypes such that mtype_to_scitype(element) is in the list scitype
        if list, returns this function element-wise applied
        if nested list/str object, replaces mtype str by scitype str
        if None, returns None

    Raises
    ------
    TypeError, if input is not of the type specified
    ValueError, if one of the strings is not a valid scitype string
    RuntimeError, if there is no mtype for the/some scitype string (this must be a bug)
    """
    msg = "scitype argument must be str or list of str"
    # handle the "None" case first
    if scitype is None or scitype == "None":
        raise TypeError(msg)
    # recurse if mtype is a list
    if isinstance(scitype, list):
        scitype_list = [y for x in scitype for y in scitype_to_mtype(x)]
        return scitype_list

    # checking for type. Checking str is enough, recursion above will do the rest.
    if not isinstance(scitype, str):
        raise TypeError(msg)

    # now we know scitype is a string, check if it is in the register
    if scitype not in SCITYPE_LIST:
        raise ValueError(
            f'"{scitype}" is not a valid scitype string, see datatypes.SCITYPE_REGISTER'
        )

    mtypes = [k[0] for k in MTYPE_REGISTER if k[1] == scitype]

    if len(mtypes) == 0:
        # if there are no mtypes, this must have been reached by mistake/bug
        raise RuntimeError("no mtypes defined for scitype " + scitype)

    if softdeps not in ["exclude", "present"]:
        return mtypes

    if softdeps == "exclude":
        # subset to mtypes that require no soft deps
        mtypes = [m for m in mtypes if m not in MTYPE_SOFT_DEPS.keys()]
        return mtypes

    if softdeps == "present":
        from sktime.utils.dependencies import _check_soft_dependencies

        def present(x):
            """Return True if x has satisfied soft dependency or has no soft dep."""
            if x not in MTYPE_SOFT_DEPS.keys():
                return True
            else:
                return _check_soft_dependencies(MTYPE_SOFT_DEPS[x], severity="none")

        # return only mtypes with soft dependencies present (or requiring none)
        mtypes = [m for m in mtypes if present(m)]
        return mtypes
