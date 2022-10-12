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
        "is_univariate": bool, True iff series has one variable
        "is_empty": bool, True iff series has no variables or no instances
        "has_nans": bool, True iff the series contains NaN values
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

check_dict = dict()


def check_pred_quantiles_proba(obj, return_metadata=False, var_name="obj"):

    metadata = dict()

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    # check if the input is a dataframe
    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} should be a pd.DataFrame"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1

    # check that column indices are unique
    if not len(set(obj.columns)) == len(obj.columns):
        msg = "column indices must be unique"
        return ret(False, msg, None, return_metadata)

    # check that all cols are numeric
    if not np.all([is_numeric_dtype(obj[c]) for c in obj.columns]):
        msg = (
            f"{var_name} should only have numeric dtype columns, "
            f"but found dtypes {obj.dtypes}"
        )
        return ret(False, msg, None, return_metadata)

    # Check time index is ordered in time
    if not index.is_monotonic_increasing:
        msg = (
            f"The (time) index of {var_name} must be sorted monotonically increasing, "
            f"but found: {index}"
        )
        return ret(False, msg, None, return_metadata)

    # check column multiindex
    colidx = obj.columns
    msg = (
        f"column of {var_name} must be pd.MultiIndex, with two levels."
        "first level is variable name, "
        "second level are (numeric) alpha values between 0 and 1."
    )

    if not isinstance(colidx, pd.MultiIndex) or not colidx.nlevels == 2:
        return ret(False, msg, None, return_metadata)
    alphas = colidx.get_level_values(1)
    if not is_numeric_dtype(alphas):
        return ret(False, msg, None, return_metadata)
    if not (alphas <= 1).all() or not (alphas >= 0).all():
        return ret(False, msg, None, return_metadata)

    # compute more metadata, only if needed
    if return_metadata:
        metadata["has_nans"] = obj.isna().values.any()
        metadata["is_univariate"] = len(colidx.get_level_values(0).unique()) == 1

    return ret(True, None, metadata, return_metadata)


check_dict[("pred_quantiles", "Proba")] = check_pred_quantiles_proba


def check_pred_interval_proba(obj, return_metadata=False, var_name="obj"):

    metadata = dict()

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    # check if the input is a dataframe
    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} should be a pd.DataFrame"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1

    # check that column indices are unique
    if not len(set(obj.columns)) == len(obj.columns):
        msg = "column indices must be unique"
        return ret(False, msg, None, return_metadata)

    # check that all cols are numeric
    if not np.all([is_numeric_dtype(obj[c]) for c in obj.columns]):
        msg = (
            f"{var_name} should only have numeric dtype columns, "
            f"but found dtypes {obj.dtypes}"
        )
        return ret(False, msg, None, return_metadata)

    # Check time index is ordered in time
    if not index.is_monotonic_increasing:
        msg = (
            f"The (time) index of {var_name} must be sorted monotonically increasing, "
            f"but found: {index}"
        )
        return ret(False, msg, None, return_metadata)

    # check column multiindex
    colidx = obj.columns
    msg = (
        f"column of {var_name} must be pd.MultiIndex, with three levels."
        "first level is variable name, "
        "second level are (numeric) coverage values between 0 and 1, "
        'third level is string "lower" or "upper", for lower/upper interval end.'
    )

    if not isinstance(colidx, pd.MultiIndex) or not colidx.nlevels == 3:
        return ret(False, msg, None, return_metadata)
    coverages = colidx.get_level_values(1)
    if not is_numeric_dtype(coverages):
        return ret(False, msg, None, return_metadata)
    if not (coverages <= 1).all() or not (coverages >= 0).all():
        return ret(False, msg, None, return_metadata)
    upper_lower = colidx.get_level_values(2)
    if not upper_lower.isin(["upper", "lower"]).all():
        return ret(False, msg, None, return_metadata)

    # compute more metadata, only if needed
    if return_metadata:
        metadata["has_nans"] = obj.isna().values.any()
        metadata["is_univariate"] = len(colidx.get_level_values(0).unique()) == 1

    return ret(True, None, metadata, return_metadata)


check_dict[("pred_interval", "Proba")] = check_pred_interval_proba
