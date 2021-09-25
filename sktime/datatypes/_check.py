# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Machine type checkers for scitypes.

Exports
-------
check_is(obj, mtype: str, scitype: str)
    checks whether obj is mtype for scitype
    returns boolean yes/no and metadata

check_raise(obj, mtype: str, scitype:str)
    checks whether obj is mtype for scitype
    returns True if passes, otherwise raises error

mtype(obj, as_scitype: str = None)
    infer the mtype of obj, considering it as as_scitype
"""

__author__ = ["fkiraly"]

__all__ = [
    "check_is",
    "check_raise",
    "mtype",
]

import numpy as np

from typing import Union, List
from sktime.datatypes._panel import check_dict_Panel
from sktime.datatypes._series import check_dict_Series
from sktime.datatypes._registry import mtype_to_scitype
from sktime.datatypes.types import Mtype, SciType

# pool convert_dict-s
check_dict = dict()
check_dict.update(check_dict_Series)
check_dict.update(check_dict_Panel)


def _check_scitype_valid(scitype: str = None):
    """Check validity of scitype."""
    valid_scitypes = list(set([x[1] for x in check_dict.keys()]))

    if scitype is not None and scitype not in valid_scitypes:
        raise TypeError(scitype + " is not a supported scitype")


def check_is(
    obj,
    mtype: Union[Mtype, List[str], List[Mtype]],
    scitype: SciType = None,
    return_metadata=False,
    var_name="obj",
):
    """Check object for compliance with mtype specification, return metadata.

    Parameters
    ----------
    obj - object to check
    mtype: str or SeriesMtype enum or PanelMtype enum or List[str] or List PanelMtype
            or SeriesMtype
        Type to check
    scitype: str or Scitype enum optional,  default = inferred from mtype
        if inferred from mtype, list elements of mtype need not have same scitype
    return_metadata - bool, optional, default=False
        if False, returns only "valid" return
        if True, returns all three return objects
    var_name: str, optional, default="obj" - name of input in error messages

    Returns
    -------
    valid: bool - whether obj is a valid object of mtype/scitype
    msg: str or list of str - error messages if object is not valid, otherwise None
            str if mtype is str; list of len(mtype) with message per mtype if list
            returned only if return_metadata is True
    metadata: dict - metadata about obj if valid, otherwise None
            returned only if return_metadata is True
        fields:
            "is_univariate": bool, True iff series has one variable
            "is_equally_spaced": bool, True iff series index is equally spaced
            "mtype": str, mtype of obj if inferred

    Raises
    ------
    TypeError if no checks defined for mtype/scitype combination
    ValueError if mtype input argument is not of expected type
    """
    if isinstance(mtype, list):
        temp = []
        for val in mtype:
            temp.append(str(val))
        mtype = temp
    else:
        if mtype is not None:
            mtype = str(mtype)

    if scitype is not None:
        scitype = str(scitype)

    _check_scitype_valid(scitype)

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    if isinstance(mtype, str):
        mtype = [mtype]
    elif isinstance(mtype, list):
        if not np.all([isinstance(x, str) for x in mtype]):
            raise ValueError("list must be a string or list of strings")
    else:
        raise ValueError("mtype must be a string or list of strings")

    valid_keys = check_dict.keys()

    # we loop through individual mtypes in mtype and see whether they pass the check
    #  for each check we remember whether it passed and what it returned
    msg = []
    found_mtype = []

    for m in mtype:
        if scitype is None:
            scitype = mtype_to_scitype(m)
        key = (m, scitype)
        if (m, scitype) not in valid_keys:
            raise TypeError(f"no check defined for mtype {m}, scitype {scitype}")

        res = check_dict[key](obj, return_metadata=return_metadata, var_name=var_name)

        if return_metadata:
            check_passed = res[0]
        else:
            check_passed = res

        if check_passed:
            found_mtype.append(m)
            final_result = res
        elif return_metadata:
            msg.append(res[1])

    # there are three options on the result of check_is:
    # a. two or more mtypes are found - this is unexpected and an error with checks
    if len(found_mtype) > 1:
        raise TypeError(
            f"Error in check_is, more than one mtype identified: {found_mtype}"
        )
    # b. one mtype is found - then return that mtype
    elif len(found_mtype) == 1:
        if return_metadata:
            # add the mtype return to the metadata
            final_result[2]["mtype"] = found_mtype[0]
            # final_result already has right shape and dependency on return_metadata
            return final_result
        else:
            return True
    # c. no mtype is found - then return False and all error messages if requested
    else:
        if len(msg) == 1:
            msg = msg[0]

        return ret(False, msg, None, return_metadata)


def check_raise(obj, mtype: Mtype, scitype: SciType = None, var_name: str = "input"):
    """Check object for compliance with mtype specification, raise errors.

    Parameters
    ----------
    obj: Any
        object to check
    mtype: str or SeriesMtype enum or PanelMtype enum or List[str] or List PanelMtype
            or SeriesMtype
        Type to check
    scitype: str or Scitype enum optional,  default = inferred from mtype
        if inferred from mtype, list elements of mtype need not have same scitype
    var_name: str, optional, default="input"
        name of input in error messages

    Returns
    -------
    valid: bool - True if obj complies with the specification
            same as when return argument of check_is is True
            otherwise raises an error

    Raises
    ------
    TypeError with informative message if obj does not comply
    TypeError if no checks defined for mtype/scitype combination
    ValueError if mtype input argument is not of expected type
    """
    if mtype is not None:
        mtype = str(mtype)
    if scitype is not None:
        scitype = str(scitype)
    obj_long_name_for_avoiding_linter_clash = obj
    valid, msg, _ = check_is(
        obj=obj_long_name_for_avoiding_linter_clash,
        mtype=mtype,
        scitype=scitype,
        return_metadata=True,
        var_name=var_name,
    )

    if valid:
        return True
    else:
        raise TypeError(msg)


def mtype(obj, as_scitype: SciType = None):
    """Infer the mtype of an object considered as a specific scitype.

    Parameters
    ----------
    obj : object to infer type of - any type, should comply with and mtype spec
        if as_scitype is provided, this needs to be mtype belonging to scitype
    as_scitype : str or Scitype enum, optional, defaults=None
        name of scitype the object "obj" is considered as, finds mtype for that
        if None (default), does not assume a specific as_scitype and tests all mtypes
            generally, as_scitype should be provided for maximum efficiency

    Returns
    -------
    str - the inferred mtype of "obj", a valid mtype string
        or None, if obj is None

    Raises
    ------
    TypeError if no type can be identified, or more than one type is identified
    """
    if obj is None:
        return None

    if as_scitype is not None:
        as_scitype = str(as_scitype)
    valid_as_scitypes = list(set([x[1] for x in check_dict.keys()]))

    if as_scitype is not None and as_scitype not in valid_as_scitypes:
        raise TypeError(as_scitype + " is not a supported scitype")
    _check_scitype_valid(as_scitype)

    if as_scitype is None:
        mtypes = np.array([x[0] for x in check_dict.keys()])
    else:
        mtypes = np.array([x[0] for x in check_dict.keys() if x[1] == as_scitype])

    res = [mtype for mtype in mtypes if check_is(obj, mtype=mtype, scitype=as_scitype)]

    if len(res) > 1:
        raise TypeError(f"Error in check_is, more than one mtype identified: {res}")

    if len(res) < 1:
        raise TypeError("No valid mtype could be identified")

    return res[0]
