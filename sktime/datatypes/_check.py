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

from sktime.datatypes._alignment import check_dict_Alignment
from sktime.datatypes._common import _metadata_requested, _ret
from sktime.datatypes._hierarchical import check_dict_Hierarchical
from sktime.datatypes._panel import check_dict_Panel
from sktime.datatypes._proba import check_dict_Proba
from sktime.datatypes._registry import AMBIGUOUS_MTYPES, SCITYPE_LIST, mtype_to_scitype
from sktime.datatypes._series import check_dict_Series
from sktime.datatypes._table import check_dict_Table

# pool convert_dict-s
check_dict = dict()
check_dict.update(check_dict_Series)
check_dict.update(check_dict_Panel)
check_dict.update(check_dict_Hierarchical)
check_dict.update(check_dict_Alignment)
check_dict.update(check_dict_Table)
check_dict.update(check_dict_Proba)


def _check_scitype_valid(scitype: str = None):
    """Check validity of scitype."""
    valid_scitypes = list({x[1] for x in check_dict.keys()})

    if not isinstance(scitype, str):
        raise TypeError(f"scitype should be a str but found {type(scitype)}")

    if scitype is not None and scitype not in valid_scitypes:
        raise TypeError(scitype + " is not a supported scitype")


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
        valid mtype strings are in datatypes.MTYPE_REGISTER (1st column)
    scitype: str, optional, scitype to check obj as; default = inferred from mtype
        if inferred from mtype, list elements of mtype need not have same scitype
        valid mtype strings are in datatypes.SCITYPE_REGISTER (1st column)
    return_metadata - bool, str, or list of str, optional, default=False
        if False, returns only "valid" return
        if True, returns all three return objects
        if str, list of str, metadata return dict is subset to keys in return_metadata
    var_name: str, optional, default="obj" - name of input in error messages

    Returns
    -------
    valid: bool - whether obj is a valid object of mtype/scitype
    msg: str or list of str - error messages if object is not valid, otherwise None
            str if mtype is str; list of len(mtype) with message per mtype if list
            returned only if return_metadata is True or str, list of str
    metadata: dict - metadata about obj if valid, otherwise None
            returned only if return_metadata is True or str, list of str
        Keys populated depend on (assumed, otherwise identified) scitype of obj.
        Always returned:
            "mtype": str, mtype of obj (assumed or inferred)
            "scitype": str, scitype of obj (assumed or inferred)
        For scitype "Series":
            "is_univariate": bool, True iff series has one variable
            "is_equally_spaced": bool, True iff series index is equally spaced
            "is_empty": bool, True iff series has no variables or no instances
            "has_nans": bool, True iff the series contains NaN values
        For scitype "Panel":
            "is_univariate": bool, True iff all series in panel have one variable
            "is_equally_spaced": bool, True iff all series indices are equally spaced
            "is_equal_length": bool, True iff all series in panel are of equal length
            "is_empty": bool, True iff one or more of the series in the panel are empty
            "is_one_series": bool, True iff there is only one series in the panel
            "has_nans": bool, True iff the panel contains NaN values
            "n_instances": int, number of instances in the panel
        For scitype "Table":
            "is_univariate": bool, True iff table has one variable
            "is_empty": bool, True iff table has no variables or no instances
            "has_nans": bool, True iff the panel contains NaN values
            "n_instances": int, number of instances/rows in the table
        For scitype "Alignment":
            currently none

    Raises
    ------
    TypeError if no checks defined for mtype/scitype combination
    TypeError if mtype input argument is not of expected type
    """
    mtype = _coerce_list_of_str(mtype, var_name="mtype")

    valid_keys = check_dict.keys()

    # we loop through individual mtypes in mtype and see whether they pass the check
    #  for each check we remember whether it passed and what it returned
    msg = []
    found_mtype = []
    found_scitype = []

    for m in mtype:
        if scitype is None:
            scitype_of_m = mtype_to_scitype(m)
        else:
            _check_scitype_valid(scitype)
            scitype_of_m = scitype
        key = (m, scitype_of_m)
        if (m, scitype_of_m) not in valid_keys:
            raise TypeError(f"no check defined for mtype {m}, scitype {scitype_of_m}")

        res = check_dict[key](obj, return_metadata=return_metadata, var_name=var_name)

        if _metadata_requested(return_metadata):
            check_passed = res[0]
        else:
            check_passed = res

        if check_passed:
            found_mtype.append(m)
            found_scitype.append(scitype_of_m)
            final_result = res
        elif _metadata_requested(return_metadata):
            msg.append(res[1])

    # there are three options on the result of check_is_mtype:
    # a. two or more mtypes are found - this is unexpected and an error with checks
    if len(found_mtype) > 1:
        raise TypeError(
            f"Error in check_is_mtype, more than one mtype identified: {found_mtype}"
        )
    # b. one mtype is found - then return that mtype
    elif len(found_mtype) == 1:
        if _metadata_requested(return_metadata):
            # add the mtype return to the metadata
            final_result[2]["mtype"] = found_mtype[0]
            final_result[2]["scitype"] = found_scitype[0]
            # final_result already has right shape and dependency on return_metadata
            return final_result
        else:
            return True
    # c. no mtype is found - then return False and all error messages if requested
    else:
        if len(msg) == 1:
            msg = msg[0]

        return _ret(False, msg, None, return_metadata)


def check_raise(obj, mtype: str, scitype: str = None, var_name: str = "input"):
    """Check object for compliance with mtype specification, raise errors.

    Parameters
    ----------
    obj - object to check
    mtype: str or list of str, mtype to check obj as
        valid mtype strings are in datatypes.MTYPE_REGISTER (1st column)
    scitype: str, optional, scitype to check obj as; default = inferred from mtype
        if inferred from mtype, list elements of mtype need not have same scitype
        valid mtype strings are in datatypes.SCITYPE_REGISTER (1st column)
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
        return_metadata=[],
        var_name=var_name,
    )

    if valid:
        return True
    else:
        raise TypeError(msg)


def mtype(
    obj,
    as_scitype: Union[str, List[str]] = None,
    exclude_mtypes=AMBIGUOUS_MTYPES,
):
    """Infer the mtype of an object considered as a specific scitype.

    Parameters
    ----------
    obj : object to infer type of - any type, should comply with some mtype spec
        if as_scitype is provided, this needs to be mtype belonging to scitype
    as_scitype : str, list of str, or None, optional, default=None
        name of scitype(s) the object "obj" is considered as, finds mtype for that
        if None (default), does not assume a specific as_scitype and tests all mtypes
            generally, as_scitype should be provided for maximum efficiency
        valid scitype type strings are in datatypes.SCITYPE_REGISTER (1st column)
    exclude_mtypes : list of str, default = AMBIGUOUS_MTYPES
        which mtypes to ignore in inferring mtype, default = ambiguous ones

    Returns
    -------
    str - the inferred mtype of "obj", a valid mtype string
            or None, if obj is None
        mtype strings with explanation are in datatypes.MTYPE_REGISTER

    Raises
    ------
    TypeError if no type can be identified, or more than one type is identified
    """
    if obj is None:
        return None

    if as_scitype is not None:
        as_scitype = _coerce_list_of_str(as_scitype, var_name="as_scitype")
        for scitype in as_scitype:
            _check_scitype_valid(scitype)

    m_plus_scitypes = [
        (x[0], x[1]) for x in check_dict.keys() if x[0] not in exclude_mtypes
    ]

    if as_scitype is not None:
        m_plus_scitypes = [(x[0], x[1]) for x in m_plus_scitypes if x[1] in as_scitype]

    # collects mtypes that are tested as valid for obj
    mtypes_positive = []

    # collects error messages from mtypes that are tested as invalid for obj
    mtypes_negative = dict()

    for m_plus_scitype in m_plus_scitypes:
        valid, msg, _ = check_is_mtype(
            obj,
            mtype=m_plus_scitype[0],
            scitype=m_plus_scitype[1],
            return_metadata=[],
        )
        if valid:
            mtypes_positive += [m_plus_scitype[0]]
        else:
            mtypes_negative[m_plus_scitype[0]] = msg

    if len(mtypes_positive) > 1:
        raise TypeError(
            f"Error in check_is_mtype, more than one mtype identified:"
            f" {mtypes_positive}"
        )

    if len(mtypes_positive) < 1:
        msg = ""
        for mtype, error in mtypes_negative.items():
            msg += f"{mtype}: {error}\r\n"
        msg = (
            f"No valid mtype could be identified for object of type {type(obj)}. "
            f"Errors returned are as follows, in format [mtype]: [error message] \r\n"
        ) + msg
        raise TypeError(msg)

    return mtypes_positive[0]


def check_is_scitype(
    obj,
    scitype: Union[str, List[str]],
    return_metadata=False,
    var_name="obj",
    exclude_mtypes=AMBIGUOUS_MTYPES,
):
    """Check object for compliance with scitype specification, return metadata.

    Parameters
    ----------
    obj - object to check
    scitype: str or list of str, scitype to check obj as
        valid mtype strings are in datatypes.SCITYPE_REGISTER
    return_metadata - bool, optional, default=False
        if False, returns only "valid" return
        if True, returns all three return objects
        if str, list of str, metadata return dict is subset to keys in return_metadata
    var_name: str, optional, default="obj" - name of input in error messages
    exclude_mtypes : list of str, default = AMBIGUOUS_MTYPES
        which mtypes to ignore in inferring mtype, default = ambiguous ones

    Returns
    -------
    valid: bool - whether obj is a valid object of mtype/scitype
    msg: dict[str, str] or None
        error messages if object is not valid, otherwise None
        keys are all mtypes tested, value for key is error message for that key
    metadata: dict - metadata about obj if valid, otherwise None
            returned only if return_metadata is True
        Fields depend on scitpe.
        Always returned:
            "mtype": str, mtype of obj (assumed or inferred)
                mtype strings with explanation are in datatypes.MTYPE_REGISTER
            "scitype": str, scitype of obj (assumed or inferred)
                scitype strings with explanation are in datatypes.SCITYPE_REGISTER
        For scitype "Series":
            "is_univariate": bool, True iff series has one variable
            "is_equally_spaced": bool, True iff series index is equally spaced
            "is_empty": bool, True iff series has no variables or no instances
            "has_nans": bool, True iff the series contains NaN values
        For scitype "Panel":
            "is_univariate": bool, True iff all series in panel have one variable
            "is_equally_spaced": bool, True iff all series indices are equally spaced
            "is_equal_length": bool, True iff all series in panel are of equal length
            "is_empty": bool, True iff one or more of the series in the panel are empty
            "is_one_series": bool, True iff there is only one series in the panel
            "has_nans": bool, True iff the panel contains NaN values
            "n_instances": int, number of instances in the panel
        For scitype "Table":
            "is_univariate": bool, True iff table has one variable
            "is_empty": bool, True iff table has no variables or no instances
            "has_nans": bool, True iff the panel contains NaN values
        For scitype "Alignment":
            currently none
    Raises
    ------
    TypeError if scitype input argument is not of expected type
    """
    scitype = _coerce_list_of_str(scitype, var_name="scitype")

    for x in scitype:
        _check_scitype_valid(x)

    valid_keys = check_dict.keys()

    # find all the mtype keys corresponding to the scitypes
    keys = [x for x in valid_keys if x[1] in scitype and x[0] not in exclude_mtypes]

    # storing the msg return
    msg = {}
    found_mtype = []
    found_scitype = []

    for key in keys:
        res = check_dict[key](obj, return_metadata=return_metadata, var_name=var_name)

        if _metadata_requested(return_metadata):
            check_passed = res[0]
        else:
            check_passed = res

        if check_passed:
            final_result = res
            found_mtype.append(key[0])
            found_scitype.append(key[1])
        elif _metadata_requested(return_metadata):
            msg[key[0]] = res[1]

    # there are three options on the result of check_is_mtype:
    # a. two or more mtypes are found - this is unexpected and an error with checks
    if len(found_mtype) > 1:
        raise TypeError(
            f"Error in check_is_mtype, more than one mtype identified: {found_mtype}"
        )
    # b. one mtype is found - then return that mtype
    elif len(found_mtype) == 1:
        if _metadata_requested(return_metadata):
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
        return _ret(False, msg, None, return_metadata)


def scitype(obj, candidate_scitypes=SCITYPE_LIST, exclude_mtypes=AMBIGUOUS_MTYPES):
    """Infer the scitype of an object.

    Parameters
    ----------
    obj : object to infer type of - any type, should comply with some mtype spec
        if as_scitype is provided, this needs to be mtype belonging to scitype
    candidate_scitypes: str or list of str, scitypes to pick from
        valid scitype strings are in datatypes.SCITYPE_REGISTER
    exclude_mtypes : list of str, default = AMBIGUOUS_MTYPES
        which mtypes to ignore in inferring mtype, default = ambiguous ones
        valid mtype strings are in datatypes.MTYPE_REGISTER

    Returns
    -------
    str - the inferred sciype of "obj", a valid scitype string
            or None, if obj is None
        scitype strings with explanation are in datatypes.SCITYPE_REGISTER

    Raises
    ------
    TypeError if no type can be identified, or more than one type is identified
    """
    candidate_scitypes = _coerce_list_of_str(
        candidate_scitypes, var_name="candidate_scitypes"
    )

    valid_scitypes = []

    for scitype in candidate_scitypes:
        valid = check_is_scitype(
            obj,
            scitype=scitype,
            return_metadata=False,
            exclude_mtypes=exclude_mtypes,
        )
        if valid:
            valid_scitypes += [scitype]

    if len(valid_scitypes) > 1:
        raise TypeError(
            "Error in function scitype, more than one valid scitype identified:"
            f"{ valid_scitypes}"
        )
    if len(valid_scitypes) == 0:
        raise TypeError(
            "Error in function scitype, no valid scitype could be identified."
        )

    return valid_scitypes[0]
