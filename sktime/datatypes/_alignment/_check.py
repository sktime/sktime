"""Machine type checkers for Alignment scitype.

Exports checkers for Alignment scitype:

check_dict: dict indexed by pairs of str
  1st element = mtype - str
  2nd element = scitype - str
elements are checker/validation functions for mtype

Function signature of all elements
check_dict[(mtype, scitype)]

Parameters
----------
obj - object to check
return_metadata - bool, optional, default=False
    if False, returns only "valid" return
    if True, returns all three return objects
    if str, list of str, metadata return dict is subset to keys in return_metadata
var_name: str, optional, default="obj" - name of input in error messages

Returns
-------
valid: bool - whether obj is a valid object of mtype/scitype
msg: str - error message if object is not valid, otherwise None
        returned only if return_metadata is True
metadata: dict - metadata about obj if valid, otherwise None
        returned only if return_metadata is True
    fields:
        currently none, placeholder
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

from sktime.datatypes._common import _ret

check_dict = dict()


def check_align(align_df, name="align_df", index="iloc"):
    """Check whether the object is a data frame in alignment format.

    Parameters
    ----------
    align_df : any object
        check passes if it follows alignment format, as follows:
        pandas.DataFrame with column names 'ind'+str(i) for integers i, as follows
            all integers i between 0 and some natural number n must be present
    name : string, optional, default="align_df"
        variable name that is printed in ValueError-s
    index : string, optional, one of "iloc" (default), "loc", "either"
        whether alignment to check is "loc" or "iloc"

    Returns
    -------
    valid : boolean, whether align_df is a valid alignment data frame
    msg : error message if align_df is invalid
    """
    if not isinstance(align_df, pd.DataFrame):
        msg = f"{name} is not a pandas DataFrame"
        return False, msg

    cols = align_df.columns
    n = len(cols)

    correctcols = {f"ind{i}" for i in range(n)}

    if not set(cols) == set(correctcols):
        msg = f"{name} index columns must be named 'ind0', 'ind1', ... 'ind{n}'"
        return False, msg

    if index == "iloc":
        # checks whether df columns are of integer (numpy or pandas nullable) type
        dtypearr = np.array([str(x) for x in align_df[cols].dtypes])
        allowedtypes = np.array(
            [
                "int",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "UInt8",
                "UInt16",
                "UInt32",
                "UInt64",
            ]
        )
        if not np.all(np.isin(dtypearr, allowedtypes)):
            msg = f"columns of {name} must have dtype intX, uintX, IntX, or UIntX"
            return False, msg
    # no additional restrictions apply if loc or either, so no elif

    return True, ""


def check_alignment_alignment(obj, return_metadata=False, var_name="obj"):
    """Check whether object has mtype `alignment` for scitype `Alignment`."""
    valid, msg = check_align(obj, name=var_name, index="iloc")

    return _ret(valid, msg, {}, return_metadata)


check_dict[("alignment", "Alignment")] = check_alignment_alignment


def check_alignment_loc_alignment(obj, return_metadata=False, var_name="obj"):
    """Check whether object has mtype `alignment_loc` for scitype `Alignment`."""
    valid, msg = check_align(obj, name=var_name, index="loc")

    return _ret(valid, msg, {}, return_metadata)


check_dict[("alignment_loc", "Alignment")] = check_alignment_loc_alignment
