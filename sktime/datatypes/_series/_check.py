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
        "is_equally_spaced": bool, True iff series index is equally spaced
        "is_empty": bool, True iff series has no variables or no instances
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

VALID_INDEX_TYPES = (pd.Int64Index, pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)

# whether the checks insist on freq attribute is set
FREQ_SET_CHECK = False


check_dict = dict()


def check_pdDataFrame_Series(obj, return_metadata=False, var_name="obj"):

    metadata = dict()

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pandas.DataFrame, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    metadata["is_univariate"] = len(obj.columns) < 2

    # check whether the time index is of valid type
    if not type(index) in VALID_INDEX_TYPES:
        msg = (
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} instead."
        )
        return ret(False, msg, None, return_metadata)

    # check that no dtype is object
    if "object" in obj.dtypes.values:
        msg = f"{var_name} should not have column of 'object' dtype"
        return ret(False, msg, None, return_metadata)

    # Check time index is ordered in time
    if not index.is_monotonic:
        msg = (
            f"The (time) index of {var_name} must be sorted monotonically increasing, "
            f"but found: {index}"
        )
        return ret(False, msg, None, return_metadata)

    if FREQ_SET_CHECK and isinstance(index, pd.DatetimeIndex):
        if index.freq is None:
            msg = f"{var_name} has DatetimeIndex, but no freq attribute set."
            return ret(False, msg, None, return_metadata)

    # check whether index is equally spaced, compute only if needed
    if return_metadata:
        metadata["is_equally_spaced"] = _index_equally_spaced(index)

    return ret(True, None, metadata, return_metadata)


check_dict[("pd.DataFrame", "Series")] = check_pdDataFrame_Series


def check_pdSeries_Series(obj, return_metadata=False, var_name="obj"):

    metadata = dict()

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    if not isinstance(obj, pd.Series):
        msg = f"{var_name} must be a pandas.Series, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a pd.Series
    index = obj.index
    metadata["is_empty"] = len(index) < 1
    metadata["is_univariate"] = True

    # check that dtype is not object
    if "object" == obj.dtypes:
        msg = f"{var_name} should not be of 'object' dtype"
        return ret(False, msg, None, return_metadata)

    # check whether the time index is of valid type
    if not type(index) in VALID_INDEX_TYPES:
        msg = (
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} instead."
        )
        return ret(False, msg, None, return_metadata)

    # Check time index is ordered in time
    if not index.is_monotonic:
        msg = (
            f"The (time) index of {var_name} must be sorted monotonically increasing, "
            f"but found: {index}"
        )
        return ret(False, msg, None, return_metadata)

    if FREQ_SET_CHECK and isinstance(index, pd.DatetimeIndex):
        if index.freq is None:
            msg = f"{var_name} has DatetimeIndex, but no freq attribute set."
            return ret(False, msg, None, return_metadata)

    # check whether index is equally spaced, compute only if needed
    if return_metadata:
        metadata["is_equally_spaced"] = _index_equally_spaced(index)

    return ret(True, None, metadata, return_metadata)


check_dict[("pd.Series", "Series")] = check_pdSeries_Series


def check_numpy_Series(obj, return_metadata=False, var_name="obj"):

    metadata = dict()

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    if len(obj.shape) == 2:
        # we now know obj is a 2D np.ndarray
        metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
        metadata["is_univariate"] = obj.shape[1] < 2
    elif len(obj.shape) == 1:
        # we now know obj is a 1D np.ndarray
        metadata["is_empty"] = len(obj) < 1
        metadata["is_univariate"] = True
    else:
        msg = f"{var_name} must be 1D or 2D numpy.ndarray, but found {len(obj.shape)}D"
        return ret(False, msg, None, return_metadata)

    # np.arrays are considered equally spaced by assumption
    metadata["is_equally_spaced"] = True

    return ret(True, None, metadata, return_metadata)


check_dict[("np.ndarray", "Series")] = check_numpy_Series


def _index_equally_spaced(index):
    """Check whether pandas.index is equally spaced.

    Parameters
    ----------
    index: pandas.Index. Must be one of:
        pd.Int64Index, pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex

    Returns
    -------
    equally_spaced: bool - whether index is equally spaced
    """
    if not isinstance(index, VALID_INDEX_TYPES):
        raise TypeError(f"index must be one of {VALID_INDEX_TYPES}")

    # empty and single element indices are equally spaced
    if len(index) < 2:
        return True

    # RangeIndex is always equally spaced
    if isinstance(index, pd.RangeIndex):
        return True

    diffs = np.diff(index)
    all_equal = np.all(diffs == diffs[0])

    return all_equal
