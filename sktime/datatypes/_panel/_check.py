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
        "is_one_series": bool, True iff there is only one series in the panel
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

from sktime.datatypes._series._check import check_pdDataFrame_Series

VALID_INDEX_TYPES = (pd.Int64Index, pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)
VALID_MULTIINDEX_TYPES = (pd.Int64Index, pd.RangeIndex)


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
        msg = f"{var_name}[i] must be Series of mtype pd.DataFrame, not at i={bad_inds}"
        return ret(False, msg, None, return_metadata)

    metadata = dict()
    metadata["is_univariate"] = np.all([res[2]["is_univariate"] for res in check_res])
    metadata["is_equally_spaced"] = np.all(
        [res[2]["is_equally_spaced"] for res in check_res]
    )
    metadata["is_empty"] = np.any([res[2]["is_empty"] for res in check_res])
    metadata["is_one_series"] = len(obj) == 1

    return ret(True, None, metadata, return_metadata)


check_dict[("df-list", "Panel")] = check_dflist_Panel


def check_numpy3D_Panel(obj, return_metadata=False, var_name="obj"):
    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    if not len(obj.shape) == 3:
        msg = f"{var_name} must be a 3D numpy.ndarray, but found {len(obj.shape)}D"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a 3D np.ndarray
    metadata = dict()
    metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1 or obj.shape[2] < 1
    metadata["is_univariate"] = obj.shape[1] < 2
    # np.arrays are considered equally spaced by assumption
    metadata["is_equally_spaced"] = True
    metadata["is_one_series"] = obj.shape[0] == 1

    return ret(True, None, metadata, return_metadata)


check_dict[("numpy3D", "Panel")] = check_numpy3D_Panel


def check_pdmultiindex_Panel(obj, return_metadata=False, var_name="obj"):
    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pd.DataFrame, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    if not isinstance(obj.index, pd.MultiIndex):
        msg = f"{var_name} have a MultiIndex, found {type(obj.index)}"
        return ret(False, msg, None, return_metadata)

    nlevels = obj.index.nlevels
    if not nlevels == 2:
        msg = f"{var_name} have a MultiIndex with 2 levels, found {nlevels}"
        return ret(False, msg, None, return_metadata)

    correct_names = ["instances", "timepoints"]
    objnames = obj.index.names
    if not objnames == correct_names:
        msg = (
            f"{var_name}  must have a MultiIndex with names"
            f" {correct_names}, found {objnames}"
        )
        return ret(False, msg, None, return_metadata)

    # check instance index being integer or range index
    instind = obj.index.droplevel(1)
    if not isinstance(instind, VALID_MULTIINDEX_TYPES):
        msg = f"instance index must be {VALID_MULTIINDEX_TYPES}, found {type(instind)}"
        return ret(False, msg, None, return_metadata)

    inst_inds = np.unique(obj.index.get_level_values(0))

    check_res = [
        check_pdDataFrame_Series(obj.loc[i], return_metadata=True) for i in inst_inds
    ]
    bad_inds = [i for i in inst_inds if not check_res[i][0]]

    if len(bad_inds) > 0:
        msg = (
            f"{var_name}.loc[i] must be Series of mtype pd.DataFrame,"
            " not at i={bad_inds}"
        )
        return ret(False, msg, None, return_metadata)

    metadata = dict()
    metadata["is_univariate"] = np.all([res[2]["is_univariate"] for res in check_res])
    metadata["is_equally_spaced"] = np.all(
        [res[2]["is_equally_spaced"] for res in check_res]
    )
    metadata["is_empty"] = np.any([res[2]["is_empty"] for res in check_res])
    metadata["is_one_series"] = len(inst_inds) == 1

    return ret(True, None, metadata, return_metadata)


check_dict[("pd-multiindex", "Panel")] = check_pdmultiindex_Panel


def _cell_is_series_or_array(cell):
    return isinstance(cell, (pd.Series, np.ndarray))


def _nested_cell_mask(X):
    return X.applymap(_cell_is_series_or_array)


def are_columns_nested(X):
    """Check whether any cells have nested structure in each DataFrame column.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to check for nested data structures.

    Returns
    -------
    any_nested : bool
        If True, at least one column is nested.
        If False, no nested columns.
    """
    any_nested = _nested_cell_mask(X).any().values
    return any_nested


def is_nested_dataframe(obj, return_metadata=False, var_name="obj"):
    """Check whether the input is a nested DataFrame.

    To allow for a mixture of nested and primitive columns types the
    the considers whether any column is a nested np.ndarray or pd.Series.

    Column is consider nested if any cells in column have a nested structure.

    Parameters
    ----------
    X: Input that is checked to determine if it is a nested DataFrame.

    Returns
    -------
    bool: Whether the input is a nested DataFrame
    """

    def ret(valid, msg, metadata, return_metadata):
        if return_metadata:
            return valid, msg, metadata
        else:
            return valid

    # If not a DataFrame we know is_nested_dataframe is False
    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pd.DataFrame, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # Otherwise we'll see if any column has a nested structure in first row
    else:
        if not are_columns_nested(obj).any():
            msg = f"{var_name} entries must be pd.Series"
            return ret(False, msg, None, return_metadata)

    metadata = dict()
    metadata["is_univariate"] = True
    # metadata["is_equally_spaced"] = todo
    # metadata["is_empty"] = todo
    metadata["is_one_series"] = len(obj) == 1

    return ret(True, None, metadata, return_metadata)


check_dict[("nested_univ", "Panel")] = is_nested_dataframe
