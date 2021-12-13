# -*- coding: utf-8 -*-
"""Machine type checkers for Table scitype.

Exports checkers for Table scitype:

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
        "is_univariate": bool, True iff table has one variable
        "is_empty": bool, True iff table has no variables or no instances
        "has_nans": bool, True iff the panel contains NaN values
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

check_dict = dict()


def _ret(valid, msg, metadata, return_metadata):
    if return_metadata:
        return valid, msg, metadata
    else:
        return valid


def check_pdDataFrame_Table(obj, return_metadata=False, var_name="obj"):

    metadata = dict()

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pandas.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    metadata["is_univariate"] = len(obj.columns) < 2

    # check whether there are any nans
    #   compute only if needed
    if return_metadata:
        metadata["has_nans"] = obj.isna().values.any()

    # check that no dtype is object
    if "object" in obj.dtypes.values:
        msg = f"{var_name} should not have column of 'object' dtype"
        return _ret(False, msg, None, return_metadata)

    return _ret(True, None, metadata, return_metadata)


check_dict[("pd_DataFrame_Table", "Table")] = check_pdDataFrame_Table


def check_numpy1D_Table(obj, return_metadata=False, var_name="obj"):

    metadata = dict()

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if len(obj.shape) != 1:
        msg = f"{var_name} must be 1D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 1D np.ndarray
    metadata["is_empty"] = len(obj) < 1
    # 1D numpy arrays are considered univariate
    metadata["is_univariate"] = True
    # check whether there any nans; compute only if requested
    if return_metadata:
        metadata["has_nans"] = np.isnan(obj).any()

    return _ret(True, None, metadata, return_metadata)


check_dict[("numpy1D", "Table")] = check_numpy1D_Table


def check_numpy2D_Table(obj, return_metadata=False, var_name="obj"):

    metadata = dict()

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if len(obj.shape) != 2:
        msg = f"{var_name} must be 1D or 2D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 2D np.ndarray
    metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
    metadata["is_univariate"] = obj.shape[1] < 2
    # check whether there any nans; compute only if requested
    if return_metadata:
        metadata["has_nans"] = np.isnan(obj).any()

    return _ret(True, None, metadata, return_metadata)


check_dict[("numpy2D", "Table")] = check_numpy2D_Table
