# -*- coding: utf-8 -*-
"""Machine type checkers for Series scitype.

Exports checkers for Series scitype:

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
var_name: str, optional, default="obj" - name of input in error messages

Returns
-------
valid: bool - whether obj is a valid object of mtype/scitype
msg: str - error message if object is not valid, otherwise None
        returned only if return_metadata is True
metadata: dict - metadata about obj if valid, otherwise None
        returned only if return_metadata is True
    fields:
        "is_univariate": bool, True iff all series in panel have one variable
        "is_equally_spaced": bool, True iff all series indices are equally spaced
        "is_empty": bool, True iff one or more of the series in the panel are empty
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

from sktime.datatypes._series._check import check_pdDataFrame_Series

VALID_INDEX_TYPES = (pd.Int64Index, pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)


check_dict = dict()


def check_dflist_Panel(obj, return_metadata=False, var_name="obj"):

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    if not isinstance(obj, list):
        msg = f"{var_name} must be list of pd.DataFrame, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    n = len(obj)

    bad_inds = [i for i in range(n) if not isinstance(obj[i], pd.DataFrame)]

    if len(bad_inds) > 0:
        msg = f"{var_name}[i] must pd.DataFrame, but found other types at i={bad_inds}"
        return ret(False, msg, None, return_metadata)

    check_res = [check_pdDataFrame_Series(s, return_metadata=True) for s in obj]
    bad_inds = [i for i in range(n) if not check_res[i][0]]

    if len(bad_inds) > 0:
        msg = f"{var_name}[i] must be Sries of mtype pd.DataFrame, not at i={bad_inds}"
        return ret(False, msg, None, return_metadata)

    metadata = dict()
    metadata["is_univariate"] = np.all([res[2]["is_univariate"] for res in check_res])
    metadata["is_equally_spaced"] = np.all(
        [res[2]["is_equally_spaced"] for res in check_res]
    )
    metadata["is_empty"] = np.any([res[2]["is_empty"] for res in check_res])

    return ret(True, None, metadata, return_metadata)


check_dict[("df-list", "Panel")] = check_dflist_Panel
