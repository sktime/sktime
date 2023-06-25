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
        "is_univariate": bool, True iff all series in panel have one variable
        "is_equally_spaced": bool, True iff all series indices are equally spaced
        "is_equal_length": bool, True iff all series in panel are of equal length
        "is_empty": bool, True iff one or more of the series in the panel are empty
        "is_one_series": bool, True iff there is only one series in the panel
        "has_nans": bool, True iff the panel contains NaN values
        "n_instances": int, number of instances in the panel
"""

__author__ = ["fkiraly", "TonyBagnall"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import is_nested_object

from sktime.datatypes._common import _req, _ret
from sktime.datatypes._series._check import (
    _index_equally_spaced,
    check_pddataframe_series,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.series import is_in_valid_index_types, is_integer_index

VALID_MULTIINDEX_TYPES = (pd.RangeIndex, pd.Index)
VALID_INDEX_TYPES = (pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)


def is_in_valid_multiindex_types(x) -> bool:
    """Check that the input type belongs to the valid multiindex types."""
    return isinstance(x, VALID_MULTIINDEX_TYPES) or is_integer_index(x)


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
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = np.all(
            [res[2]["is_univariate"] for res in check_res]
        )
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = np.all(
            [res[2]["is_equally_spaced"] for res in check_res]
        )
    if _req("is_equal_length", return_metadata):
        metadata["is_equal_length"] = _list_all_equal([len(s) for s in obj])
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = np.any([res[2]["is_empty"] for res in check_res])
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = np.any([res[2]["has_nans"] for res in check_res])
    if _req("is_one_series", return_metadata):
        metadata["is_one_series"] = n == 1
    if _req("n_panels", return_metadata):
        metadata["n_panels"] = 1
    if _req("is_one_panel", return_metadata):
        metadata["is_one_panel"] = True
    if _req("n_instances", return_metadata):
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
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1 or obj.shape[2] < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = obj.shape[1] < 2
    # np.arrays are considered equally spaced and equal length by assumption
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = True
    if _req("is_equal_length", return_metadata):
        metadata["is_equal_length"] = True

    if _req("n_instances", return_metadata):
        metadata["n_instances"] = obj.shape[0]
    if _req("is_one_series", return_metadata):
        metadata["is_one_series"] = obj.shape[0] == 1
    if _req("n_panels", return_metadata):
        metadata["n_panels"] = 1
    if _req("is_one_panel", return_metadata):
        metadata["is_one_panel"] = True

    # check whether there any nans; only if requested
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = pd.isnull(obj).any()

    return _ret(True, None, metadata, return_metadata)


check_dict[("numpy3D", "Panel")] = check_numpy3d_panel


def check_pdmultiindex_panel(obj, return_metadata=False, var_name="obj", panel=True):
    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pd.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not isinstance(obj.index, pd.MultiIndex):
        msg = f"{var_name} must have a MultiIndex, found {type(obj.index)}"
        return _ret(False, msg, None, return_metadata)

    index = obj.index

    # check that columns are unique
    col_names = obj.columns
    if not col_names.is_unique:
        msg = f"{var_name} must have unique column indices, but found {col_names}"
        return _ret(False, msg, None, return_metadata)

    # check that no dtype is object
    if "object" in obj.dtypes.values:
        msg = f"{var_name} should not have column of 'object' dtype"
        return _ret(False, msg, None, return_metadata)

    # check that there are precisely two index levels
    nlevels = index.nlevels
    if panel is True and not nlevels == 2:
        msg = f"{var_name} must have a MultiIndex with 2 levels, found {nlevels}"
        return _ret(False, msg, None, return_metadata)
    elif panel is False and not nlevels > 2:
        msg = (
            f"{var_name} must have a MultiIndex with 3 or more levels, found {nlevels}"
        )
        return _ret(False, msg, None, return_metadata)

    # check whether the time index is of valid type
    if not is_in_valid_index_types(index.levels[-1]):
        msg = (
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} or integer index instead."
        )
        return _ret(False, msg, None, return_metadata)

    # check instance index being integer or range index
    inst_inds = index.levels[0]
    if not is_in_valid_multiindex_types(inst_inds):
        msg = (
            f"instance index (first/highest index) must be {VALID_MULTIINDEX_TYPES}, "
            f"integer index, but found {type(inst_inds)}"
        )
        return _ret(False, msg, None, return_metadata)

    # check if time index is monotonic increasing for each group
    if not index.is_monotonic_increasing:
        index_frame = index.to_frame()
        if (
            not index_frame.groupby(level=list(range(index.nlevels - 1)), sort=False)[
                index_frame.columns[-1]
            ]
            .is_monotonic_increasing.astype(bool)
            .all()
        ):
            msg = (
                f"The (time) index of {var_name} must be sorted monotonically "
                f"increasing, but found: {index}"
            )
            return _ret(False, msg, None, return_metadata)

    metadata = dict()

    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = len(obj.columns) < 2
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()

    # check whether index is equally spaced or if there are any nans
    #   compute only if needed
    requires_series_grps = [
        "n_instances",
        "is_one_series",
        "is_equal_length",
        "is_equally_spaced",
    ]
    if _req(requires_series_grps, return_metadata):
        series_groups = obj.groupby(level=list(range(index.nlevels - 1)), sort=False)
        n_series = series_groups.ngroups

        if _req("n_instances", return_metadata):
            metadata["n_instances"] = n_series
        if _req("is_one_series", return_metadata):
            metadata["is_one_series"] = n_series == 1
        if _req("is_equal_length", return_metadata):
            metadata["is_equal_length"] = _list_all_equal(
                series_groups.size().to_numpy()
            )
        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = all(
                _index_equally_spaced(group.index.get_level_values(-1))
                for _, group in series_groups
            )

    requires_panel_grps = ["n_panels", "is_one_panel"]
    if _req(requires_panel_grps, return_metadata):
        if panel:
            n_panels = 1
        else:
            panel_groups = obj.groupby(level=list(range(index.nlevels - 2)), sort=False)
            n_panels = panel_groups.ngroups

        if _req("n_panels", return_metadata):
            metadata["n_panels"] = n_panels
        if _req("is_one_panel", return_metadata):
            metadata["is_one_panel"] = n_panels == 1

    return _ret(True, None, metadata, return_metadata)


check_dict[("pd-multiindex", "Panel")] = check_pdmultiindex_panel


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
    any_nested = any(is_nested_object(series) for _, series in X.items())
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
        if not all([i == "object" for i in obj.dtypes]):
            msg = f"{var_name} All columns must be object, found {type(obj)}"
            return _ret(False, msg, None, return_metadata)

        if not are_columns_nested(obj):
            msg = f"{var_name} entries must be pd.Series"
            return _ret(False, msg, None, return_metadata)

    # check that columns are unique
    if not obj.columns.is_unique:
        msg = f"{var_name} must have unique column indices, but found {obj.columns}"
        return _ret(False, msg, None, return_metadata)

    # Check instance index is unique
    if not obj.index.is_unique:
        duplicates = obj.index[obj.index.duplicated()].unique().to_list()
        msg = (
            f"The instance index of {var_name} must be unique, "
            f"but found duplicates: {duplicates}"
        )
        return _ret(False, msg, None, return_metadata)

    metadata = dict()
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = obj.shape[1] < 2
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(obj)
    if _req("is_one_series", return_metadata):
        metadata["is_one_series"] = len(obj) == 1
    if _req("n_panels", return_metadata):
        metadata["n_panels"] = 1
    if _req("is_one_panel", return_metadata):
        metadata["is_one_panel"] = True
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = _nested_dataframe_has_nans(obj)
    if _req("is_equal_length", return_metadata):
        metadata["is_equal_length"] = not _nested_dataframe_has_unequal(obj)

    # todo: this is temporary override, proper is_empty logic needs to be added
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = False
    if _req("is_equally_spaced", return_metadata):
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
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = True
    # np.arrays are considered equally spaced, equal length, by assumption
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = True
    if _req("is_equal_length", return_metadata):
        metadata["is_equal_length"] = True
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = obj.shape[0]
    if _req("is_one_series", return_metadata):
        metadata["is_one_series"] = obj.shape[0] == 1
    if _req("n_panels", return_metadata):
        metadata["n_panels"] = 1
    if _req("is_one_panel", return_metadata):
        metadata["is_one_panel"] = True
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = np.isnan(obj).any()

    return _ret(True, None, metadata, return_metadata)


check_dict[("numpyflat", "Panel")] = check_numpyflat_Panel

if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import check_dask_frame

    def check_dask_panel(obj, return_metadata=False, var_name="obj"):
        return check_dask_frame(
            obj=obj,
            return_metadata=return_metadata,
            var_name=var_name,
            scitype="Panel",
        )

    check_dict[("dask_panel", "Panel")] = check_dask_panel
