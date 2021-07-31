# -*- coding: utf-8 -*-
"""Machine type checkers for scitypes.

Exports
-------
check_is(obj, mtype: str, scitype: str)
    checks whether obj is mtype for scitype
"""

__author__ = ["fkiraly"]

__all__ = [
    "check_is",
    "mtype",
]

import numpy as np

from sktime.datatypes._series import check_dict_Series

# pool convert_dict-s and infer_mtype_dict-s
check_dict = dict()
check_dict.update(check_dict_Series)
# check_dict.update(check_dict_Panel)


def check_is(obj, mtype: str, scitype: str, return_metadata=False, var_name="obj"):
    """Convert objects between different machine representations, subject to scitype.

    Parameters
    ----------
    obj - object to check
    mtype: str or list of str, mtype to check obj as
    scitype: str, scitype to check obj as
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
    """
    if isinstance(mtype, str):
        mtype = [mtype]
    elif isinstance(mtype, list):
        if not np.all([isinstance(x, str) for x in mtype]):
            raise ValueError("list must be a string or list of strings")
    else:
        raise ValueError("list must be a string or list of strings")

    valid_keys = [x for x in list(check_dict.keys()) if x[1] == scitype]

    msg = []

    for m in mtype:
        key = (m, scitype)
        if (m, scitype) not in valid_keys:
            raise ValueError(f"no check defined for mtype {m}, scitype {scitype}")

        res = check_dict[key](obj, return_metadata=return_metadata, var_name=var_name)

        if return_metadata:
            check_passed = res[0]
        else:
            check_passed = res

        if check_passed:
            return res
        elif return_metadata:
            msg.append(res[1])

    return False, msg, None


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
    TypeError if no type can be identified, or more than one type is identified
    """
    if obj is None:
        return None

    valid_as_scitypes = list(set([x[1] for x in list(check_dict.keys())]))

    if as_scitype not in valid_as_scitypes:
        raise TypeError(as_scitype + " is not a supported scitype")

    mtypes = [x[0] for x in list(check_dict.keys()) if x[1] == as_scitype]

    is_mtype = [check_is(obj, mtype=mtype, scitype=as_scitype) for mtype in mtypes]
    is_mtype = np.array(is_mtype)
    res = mtypes[is_mtype]

    if np.sum(is_mtype) > 1:
        raise TypeError(f"Error in check_is, more than one mtype identified: {res}")

    if np.sum(is_mtype) < 1:
        raise TypeError("")

    return res[0]
