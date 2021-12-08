# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Machine type checkers for scitypes.

Exports
-------
check_is_mtype(obj, mtype: str, scitype: str)
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
    "check_is_mtype",
    "check_raise",
    "mtype",
]

from typing import List, Union

import numpy as np
from deprecated.sphinx import deprecated

from sktime.datatypes._alignment import check_dict_Alignment
from sktime.datatypes._panel import check_dict_Panel
from sktime.datatypes._registry import mtype_to_scitype
from sktime.datatypes._series import check_dict_Series

# pool convert_dict-s
check_dict = dict()
check_dict.update(check_dict_Series)
check_dict.update(check_dict_Panel)
check_dict.update(check_dict_Alignment)


def _check_scitype_valid(scitype: str = None):
    """Check validity of scitype."""
    valid_scitypes = list(set([x[1] for x in check_dict.keys()]))

    if scitype is not None and scitype not in valid_scitypes:
        raise TypeError(scitype + " is not a supported scitype")


#  TODO: remove in v0.11.0
@deprecated(
    version="v0.10.0",
    reason=(
        "check_is has been deprecated and will be removed in v0.11.0."
        "Please use check_is_mtype instead."
    ),
    category=FutureWarning,
)
def check_is(
    obj,
    mtype: Union[str, List[str]],
    scitype: str = None,
    return_metadata=False,
    var_name="obj",
):
    """Check object for compliance with mtype specification, return metadata.

    Parameters
    ----------
    obj - object to check
    mtype: str or list of str, mtype to check obj as
    scitype: str, optional, scitype to check obj as; default = inferred from mtype
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
    return check_is_mtype(
        obj=obj,
        mtype=mtype,
        scitype=scitype,
        return_metadata=return_metadata,
        var_name=var_name,
    )


def _coerce_list_of_str(obj, var_name="obj"):
    """Check whether object is string or list of string.

    Parameters
    ----------
    obj - object to check
    var_name: str, optional, default="obj" - name of input in error messages

    Returns
    -------
    list of str
        equal to obj if was a list; equal to [obj] if obj was a str
        note: if obj was a list, return is not a copy, but identical

    Raises
    ------
    TypeError if obj is not a str or list of str
    """
    if isinstance(obj, str):
        obj = [obj]
    elif isinstance(obj, list):
        if not np.all([isinstance(x, str) for x in obj]):
            raise TypeError(f"{var_name} must be a string or list of strings")
    else:
        raise TypeError(f"{var_name} must be a string or list of strings")

    return obj


def check_is_mtype(
    obj,
    mtype: Union[str, List[str]],
    scitype: str = None,
    return_metadata=False,
    var_name="obj",
):
    """Check object for compliance with mtype specification, return metadata.

    Parameters
    ----------
    obj - object to check
    mtype: str or list of str, mtype to check obj as
    scitype: str, optional, scitype to check obj as; default = inferred from mtype
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
    TypeError if mtype input argument is not of expected type
    """
    _check_scitype_valid(scitype)

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    mtype = _coerce_list_of_str(mtype, var_name="mtype")

    valid_keys = check_dict.keys()

    # we loop through individual mtypes in mtype and see whether they pass the check
    #  for each check we remember whether it passed and what it returned
    msg = []
    found_mtype = []

    for m in mtype:
        if scitype is None:
            scitype_of_m = mtype_to_scitype(m)
        else:
            scitype_of_m = scitype
        key = (m, scitype_of_m)
        if (m, scitype_of_m) not in valid_keys:
            raise TypeError(f"no check defined for mtype {m}, scitype {scitype_of_m}")

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

    # there are three options on the result of check_is_mtype:
    # a. two or more mtypes are found - this is unexpected and an error with checks
    if len(found_mtype) > 1:
        raise TypeError(
            f"Error in check_is_mtype, more than one mtype identified: {found_mtype}"
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


def check_raise(obj, mtype: str, scitype: str = None, var_name: str = "input"):
    """Check object for compliance with mtype specification, raise errors.

    Parameters
    ----------
    obj - object to check
    mtype: str or list of str, mtype to check obj as
    scitype: str, optional, scitype to check obj as; default = inferred from mtype
        if inferred from mtype, list elements of mtype need not have same scitype
    var_name: str, optional, default="input" - name of input in error messages

    Returns
    -------
    valid: bool - True if obj complies with the specification
            same as when return argument of check_is_mtype is True
            otherwise raises an error

    Raises
    ------
    TypeError with informative message if obj does not comply
    TypeError if no checks defined for mtype/scitype combination
    ValueError if mtype input argument is not of expected type
    """
    obj_long_name_for_avoiding_linter_clash = obj
    valid, msg, _ = check_is_mtype(
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


def mtype(obj, as_scitype: str = None):
    """Infer the mtype of an object considered as a specific scitype.

    Parameters
    ----------
    obj : object to infer type of - any type, should comply with and mtype spec
        if as_scitype is provided, this needs to be mtype belonging to scitype
    as_scitype : str, optional, default=None
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

    _check_scitype_valid(as_scitype)

    if as_scitype is None:
        mtypes = np.array([x[0] for x in check_dict.keys()])
    else:
        mtypes = np.array([x[0] for x in check_dict.keys() if x[1] == as_scitype])

    res = [
        mtype
        for mtype in mtypes
        if check_is_mtype(obj, mtype=mtype, scitype=as_scitype)
    ]

    if len(res) > 1:
        raise TypeError(
            f"Error in check_is_mtype, more than one mtype identified: {res}"
        )

    if len(res) < 1:
        raise TypeError("No valid mtype could be identified")

    return res[0]


def check_is_scitype(
    obj,
    scitype: Union[str, List[str]],
    return_metadata=False,
    var_name="obj",
):
    """Check object for compliance with mtype specification, return metadata.

    Parameters
    ----------
    obj - object to check
    scitype: str or list of str, scitype to check obj as
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
            "mtype": str, mtype of obj if inferred
            "scitype": str, scitype of obj if inferred

    Raises
    ------
    TypeError if scitype input argument is not of expected type
    """

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    scitype = _coerce_list_of_str(scitype, var_name="scitype")

    for x in scitype:
        _check_scitype_valid(x)

    valid_keys = check_dict.keys()

    # find all the mtype keys corresponding to the scitypes
    keys = [x for x in valid_keys if x[1] in scitype]

    # storing the msg retursn
    msg = []
    found_mtype = []
    found_scitype = []

    for key in keys:
        res = check_dict[key](obj, return_metadata=return_metadata, var_name=var_name)

        if return_metadata:
            check_passed = res[0]
        else:
            check_passed = res

        if check_passed:
            final_result = res
            found_mtype.append([0])
            found_scitype.append(key[1])
        elif return_metadata:
            msg.append(res[1])

    # there are three options on the result of check_is_mtype:
    # a. two or more mtypes are found - this is unexpected and an error with checks
    if len(found_mtype) > 1:
        raise TypeError(
            f"Error in check_is_mtype, more than one mtype identified: {found_mtype}"
        )
    # b. one mtype is found - then return that mtype
    elif len(found_mtype) == 1:
        if return_metadata:
            # add the mtype return to the metadata
            final_result[2]["mtype"] = found_mtype[0]
            # add the scitype return to the metadata
            final_result[2]["scitype"] = found_scitype[0]
            # final_result already has right shape and dependency on return_metadata
            return final_result
        else:
            return True
    # c. no mtype is found - then return False and all error messages if requested
    else:
        if len(msg) == 1:
            msg = msg[0]

        return ret(False, msg, None, return_metadata)
