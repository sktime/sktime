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
        "is_equal_length": bool, True iff all series in panel are of equal length
        "is_empty": bool, True iff one or more of the series in the panel are empty
        "is_one_series": bool, True iff there is only one series in the panel
        "has_nans": bool, True iff the panel contains NaN values
        "n_instances": int, number of instances in the panel
"""

__author__ = ["fkiraly", "tonybagnall"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

from sktime.datatypes._series._check import check_pddataframe_series
from sktime.utils.validation.series import is_integer_index

VALID_MULTIINDEX_TYPES = (pd.RangeIndex, pd.Index)


def is_in_valid_multiindex_types(x) -> bool:
    """Check that the input type belongs to the valid multiindex types."""
    return isinstance(x, VALID_MULTIINDEX_TYPES) or is_integer_index(x)


def _ret(valid, msg, metadata, return_metadata):
    if return_metadata:
        return valid, msg, metadata
    else:
        return valid


def _list_all_equal(obj):
    """Check whether elements of list are all equal.

    Parameters
    ----------
    obj: list - assumed, not checked

    Returns
    -------
    bool, True if elements of obj are all equal
    """
    if len(obj) < 2:
        return True

    return np.all([s == obj[0] for s in obj])


check_dict = dict()


def check_dflist_panel(obj, return_metadata=False, var_name="obj"):

    if not isinstance(obj, list):
        msg = f"{var_name} must be list of pd.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    n = len(obj)

    bad_inds = [i for i in range(n) if not isinstance(obj[i], pd.DataFrame)]

    if len(bad_inds) > 0:
        msg = f"{var_name}[i] must pd.DataFrame, but found other types at i={bad_inds}"
        return _ret(False, msg, None, return_metadata)

    check_res = [check_pddataframe_series(s, return_metadata=True) for s in obj]
    bad_inds = [i for i in range(n) if not check_res[i][0]]

    if len(bad_inds) > 0:
        msg = f"{var_name}[i] must be Series of mtype pd.DataFrame, not at i={bad_inds}"
        return _ret(False, msg, None, return_metadata)

    metadata = dict()
    metadata["is_univariate"] = np.all([res[2]["is_univariate"] for res in check_res])
    metadata["is_equally_spaced"] = np.all(
        [res[2]["is_equally_spaced"] for res in check_res]
    )
    metadata["is_equal_length"] = _list_all_equal([len(s) for s in obj])
    metadata["is_empty"] = np.any([res[2]["is_empty"] for res in check_res])
    metadata["has_nans"] = np.any([res[2]["has_nans"] for res in check_res])
    metadata["is_one_series"] = n == 1
    metadata["n_instances"] = n

    return _ret(True, None, metadata, return_metadata)


check_dict[("df-list", "Panel")] = check_dflist_panel


def check_numpy3d_panel(obj, return_metadata=False, var_name="obj"):

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not len(obj.shape) == 3:
        msg = f"{var_name} must be a 3D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 3D np.ndarray
    metadata = dict()
    metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1 or obj.shape[2] < 1
    metadata["is_univariate"] = obj.shape[1] < 2
    # np.arrays are considered equally spaced and equal length by assumption
    metadata["is_equally_spaced"] = True
    metadata["is_equal_length"] = True

    metadata["n_instances"] = obj.shape[0]
    metadata["is_one_series"] = obj.shape[0] == 1

    # check whether there any nans; only if requested
    if return_metadata:
        metadata["has_nans"] = pd.isnull(obj).any()

    return _ret(True, None, metadata, return_metadata)


check_dict[("numpy3D", "Panel")] = check_numpy3d_panel


def check_pdmultiindex_panel(obj, return_metadata=False, var_name="obj"):

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pd.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not isinstance(obj.index, pd.MultiIndex):
        msg = f"{var_name} must have a MultiIndex, found {type(obj.index)}"
        return _ret(False, msg, None, return_metadata)

    # check that columns are unique
    msg = f"{var_name} must have " f"unique column indices, but found {obj.columns}"
    assert obj.columns.is_unique, msg

    # check that there are precisely two index levels
    nlevels = obj.index.nlevels
    if not nlevels == 2:
        msg = f"{var_name} must have a MultiIndex with 2 levels, found {nlevels}"
        return _ret(False, msg, None, return_metadata)

    # check instance index being integer or range index
    instind = obj.index.get_level_values(0)
    if not is_in_valid_multiindex_types(instind):
        msg = (
            f"instance index (first/highest index) must be {VALID_MULTIINDEX_TYPES}, "
            f"integer index, but found {type(instind)}"
        )
        return _ret(False, msg, None, return_metadata)

    inst_inds = obj.index.get_level_values(0).unique()
    # inst_inds = np.unique(obj.index.get_level_values(0))

    check_res = [
        check_pddataframe_series(obj.loc[i], return_metadata=True) for i in inst_inds
    ]
    bad_inds = [i for i in range(len(inst_inds)) if not check_res[i][0]]

    if len(bad_inds) > 0:
        msg = (
            f"{var_name}.loc[i] must be Series of mtype pd.DataFrame,"
            f" not at i={bad_inds}"
        )
        return _ret(False, msg, None, return_metadata)

    metadata = dict()
    metadata["is_univariate"] = np.all([res[2]["is_univariate"] for res in check_res])
    metadata["is_equally_spaced"] = np.all(
        [res[2]["is_equally_spaced"] for res in check_res]
    )
    metadata["is_empty"] = np.any([res[2]["is_empty"] for res in check_res])
    metadata["n_instances"] = len(inst_inds)
    metadata["is_one_series"] = len(inst_inds) == 1
    metadata["has_nans"] = obj.isna().values.any()
    metadata["is_equal_length"] = _list_all_equal([len(obj.loc[i]) for i in inst_inds])

    return _ret(True, None, metadata, return_metadata)


check_dict[("pd-multiindex", "Panel")] = check_pdmultiindex_panel


def _cell_is_series(cell):
    return isinstance(cell, pd.Series)


def _nested_cell_mask(X):
    return X.applymap(_cell_is_series)


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


def _nested_dataframe_has_unequal(X: pd.DataFrame) -> bool:
    """Check whether an input nested DataFrame of Series has unequal length series.

    Parameters
    ----------
    X : pd.DataFrame where each cell is a pd.Series

    Returns
    -------
    True if x contains any NaNs, False otherwise.
    """
    rows = len(X)
    cols = len(X.columns)
    s = X.iloc[0, 0]
    length = len(s)

    for i in range(0, rows):
        for j in range(0, cols):
            s = X.iloc[i, j]
            temp = len(s)
            if temp != length:
                return True
    return False


def _nested_dataframe_has_nans(X: pd.DataFrame) -> bool:
    """Check whether an input pandas of Series has nans.

    Parameters
    ----------
    X : pd.DataFrame where each cell is a pd.Series

    Returns
    -------
    True if x contains any NaNs, False otherwise.
    """
    cases = len(X)
    dimensions = len(X.columns)
    for i in range(cases):
        for j in range(dimensions):
            s = X.iloc[i, j]
            if hasattr(s, "size"):
                for k in range(s.size):
                    if pd.isna(s.iloc[k]):
                        return True
            elif pd.isna(s):
                return True
    return False


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
    # If not a DataFrame we know is_nested_dataframe is False
    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pd.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    # Otherwise we'll see if any column has a nested structure in first row
    else:
        if not are_columns_nested(obj).any():
            msg = f"{var_name} entries must be pd.Series"
            return _ret(False, msg, None, return_metadata)

    # check that columns are unique
    msg = f"{var_name} must have " f"unique column indices, but found {obj.columns}"
    assert obj.columns.is_unique, msg

    # Check instance index is unique
    if not obj.index.is_unique:
        duplicates = obj.index[obj.index.duplicated()].unique().to_list()
        msg = (
            f"The instance index of {var_name} must be unique, "
            f"but found duplicates: {duplicates}"
        )
        return _ret(False, msg, None, return_metadata)

    metadata = dict()
    metadata["is_univariate"] = obj.shape[1] < 2
    metadata["n_instances"] = len(obj)
    metadata["is_one_series"] = len(obj) == 1
    if return_metadata:
        metadata["has_nans"] = _nested_dataframe_has_nans(obj)
        metadata["is_equal_length"] = not _nested_dataframe_has_unequal(obj)

    # todo: this is temporary override, proper is_empty logic needs to be added
    metadata["is_empty"] = False
    metadata["is_equally_spaced"] = True
    # end hacks

    return _ret(True, None, metadata, return_metadata)


check_dict[("nested_univ", "Panel")] = is_nested_dataframe


def check_numpyflat_Panel(obj, return_metadata=False, var_name="obj"):

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not len(obj.shape) == 2:
        msg = f"{var_name} must be a 2D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 3D np.ndarray
    metadata = dict()
    metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
    metadata["is_univariate"] = True
    # np.arrays are considered equally spaced, equal length, by assumption
    metadata["is_equally_spaced"] = True
    metadata["is_equal_length"] = True
    metadata["n_instances"] = obj.shape[0]
    metadata["is_one_series"] = obj.shape[0] == 1

    # check whether there any nans; only if requested
    if return_metadata:
        metadata["has_nans"] = np.isnan(obj).any()

    return _ret(True, None, metadata, return_metadata)


check_dict[("numpyflat", "Panel")] = check_numpyflat_Panel
