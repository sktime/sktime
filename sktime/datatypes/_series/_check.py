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
        "is_univariate": bool, True iff series has one variable
        "is_equally_spaced": bool, True iff series index is equally spaced
        "is_empty": bool, True iff series has no variables or no instances
        "has_nans": bool, True iff the series contains NaN values
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

from sktime.datatypes._common import _req
from sktime.datatypes._common import _ret as ret
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.series import is_in_valid_index_types

VALID_INDEX_TYPES = (pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)

# whether the checks insist on freq attribute is set
FREQ_SET_CHECK = False


check_dict = dict()


def check_pddataframe_series(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pandas.DataFrame, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = len(obj.columns) < 2

    # check that columns are unique
    if not obj.columns.is_unique:
        msg = f"{var_name} must have unique column indices, but found {obj.columns}"
        return ret(False, msg, None, return_metadata)

    # check whether the time index is of valid type
    if not is_in_valid_index_types(index):
        msg = (
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} or integer index instead."
        )
        return ret(False, msg, None, return_metadata)

    # check that no dtype is object
    if "object" in obj.dtypes.values:
        msg = f"{var_name} should not have column of 'object' dtype"
        return ret(False, msg, None, return_metadata)

    # Check time index is ordered in time
    if not index.is_monotonic_increasing:
        msg = (
            f"The (time) index of {var_name} must be sorted monotonically increasing, "
            f"but found: {index}"
        )
        return ret(False, msg, None, return_metadata)

    if FREQ_SET_CHECK and isinstance(index, pd.DatetimeIndex):
        if index.freq is None:
            msg = f"{var_name} has DatetimeIndex, but no freq attribute set."
            return ret(False, msg, None, return_metadata)

    # check whether index is equally spaced or if there are any nans
    #   compute only if needed
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = _index_equally_spaced(index)
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()

    return ret(True, None, metadata, return_metadata)


check_dict[("pd.DataFrame", "Series")] = check_pddataframe_series


def check_pdseries_series(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, pd.Series):
        msg = f"{var_name} must be a pandas.Series, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a pd.Series
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = True

    # check that dtype is not object
    if "object" == obj.dtypes:
        msg = f"{var_name} should not be of 'object' dtype"
        return ret(False, msg, None, return_metadata)

    # check whether the time index is of valid type
    if not is_in_valid_index_types(index):
        msg = (
            f"{type(index)} is not supported for {var_name}, use "
            f"one of {VALID_INDEX_TYPES} or integer index instead."
        )
        return ret(False, msg, None, return_metadata)

    # Check time index is ordered in time
    if not index.is_monotonic_increasing:
        msg = (
            f"The (time) index of {var_name} must be sorted monotonically increasing, "
            f"but found: {index}"
        )
        return ret(False, msg, None, return_metadata)

    if FREQ_SET_CHECK and isinstance(index, pd.DatetimeIndex):
        if index.freq is None:
            msg = f"{var_name} has DatetimeIndex, but no freq attribute set."
            return ret(False, msg, None, return_metadata)

    # check whether index is equally spaced or if there are any nans
    #   compute only if needed
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = _index_equally_spaced(index)
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()

    return ret(True, None, metadata, return_metadata)


check_dict[("pd.Series", "Series")] = check_pdseries_series


def check_numpy_series(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    if len(obj.shape) == 2:
        # we now know obj is a 2D np.ndarray
        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
        if _req("is_univariate", return_metadata):
            metadata["is_univariate"] = obj.shape[1] < 2
    elif len(obj.shape) == 1:
        # we now know obj is a 1D np.ndarray
        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(obj) < 1
        if _req("is_univariate", return_metadata):
            metadata["is_univariate"] = True
    else:
        msg = f"{var_name} must be 1D or 2D numpy.ndarray, but found {len(obj.shape)}D"
        return ret(False, msg, None, return_metadata)

    # np.arrays are considered equally spaced by assumption
    if _req("is_equally_spaced", return_metadata):
        metadata["is_equally_spaced"] = True

    # check whether there any nans; compute only if requested
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = pd.isnull(obj).any()

    return ret(True, None, metadata, return_metadata)


check_dict[("np.ndarray", "Series")] = check_numpy_series


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
    if not is_in_valid_index_types(index):
        raise TypeError(f"index must be one of {VALID_INDEX_TYPES} or integer index")

    # empty, single and two-element indices are equally spaced
    if len(index) < 3:
        return True

    # RangeIndex is always equally spaced
    if isinstance(index, pd.RangeIndex):
        return True

    if isinstance(index, pd.PeriodIndex):
        return index.is_full

    # we now treat a necessary condition for being equally spaced:
    # the first two spaces are equal. From now on, we know this.
    if index[1] - index[0] != index[2] - index[1]:
        return False

    # another necessary condition for equally spaced:
    # index span is number of spaces times first space
    n = len(index)
    if index[n - 1] - index[0] != (n - 1) * (index[1] - index[0]):
        return False

    # fallback for all other cases:
    # in general, we need to compute all differences and check explicitly
    # CAVEAT: this has a comparabily long runtime and high memory usage
    diffs = np.diff(index)
    all_equal = np.all(diffs == diffs[0])

    return all_equal


if _check_soft_dependencies("xarray", severity="none"):
    import xarray as xr

    def check_xrdataarray_series(obj, return_metadata=False, var_name="obj"):
        metadata = {}

        if not isinstance(obj, xr.DataArray):
            msg = f"{var_name} must be a xarray.DataArray, found {type(obj)}"
            return ret(False, msg, None, return_metadata)

        # we now know obj is a xr.DataArray
        if len(obj.dims) > 2:  # Without multi indexing only two dimensions are possible
            msg = f"{var_name} must have two or less dimension, found {type(obj.dims)}"
            return ret(False, msg, None, return_metadata)

        # The first dimension is the index of the time series in sktimelen
        index = obj.indexes[obj.dims[0]]

        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(index) < 1 or len(obj.values) < 1
        # The second dimension is the set of columns
        if _req("is_univariate", return_metadata):
            metadata["is_univariate"] = len(obj.dims) == 1 or len(obj[obj.dims[1]]) < 2

        # check that columns are unique
        if not len(obj.dims) == len(set(obj.dims)):
            msg = f"{var_name} must have unique column indices, but found {obj.dims}"
            return ret(False, msg, None, return_metadata)

        # check whether the time index is of valid type
        if not is_in_valid_index_types(index):
            msg = (
                f"{type(index)} is not supported for {var_name}, use "
                f"one of {VALID_INDEX_TYPES} or integer index instead."
            )
            return ret(False, msg, None, return_metadata)

        # check that the dtype is not object
        if "object" == obj.dtype:
            msg = f"{var_name} should not have column of 'object' dtype"
            return ret(False, msg, None, return_metadata)

        # Check time index is ordered in time
        if not index.is_monotonic_increasing:
            msg = (
                f"The (time) index of {var_name} must be sorted "
                f"monotonically increasing, but found: {index}"
            )
            return ret(False, msg, None, return_metadata)

        if FREQ_SET_CHECK and isinstance(index, pd.DatetimeIndex):
            if index.freq is None:
                msg = f"{var_name} has DatetimeIndex, but no freq attribute set."
                return ret(False, msg, None, return_metadata)

        # check whether index is equally spaced or if there are any nans
        #   compute only if needed
        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = _index_equally_spaced(index)
        if _req("has_nans", return_metadata):
            metadata["has_nans"] = obj.isnull().values.any()

        return ret(True, None, metadata, return_metadata)

    check_dict[("xr.DataArray", "Series")] = check_xrdataarray_series


if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import check_dask_frame

    def check_dask_series(obj, return_metadata=False, var_name="obj"):
        return check_dask_frame(
            obj=obj,
            return_metadata=return_metadata,
            var_name=var_name,
            freq_set_check=FREQ_SET_CHECK,
            scitype="Series",
        )

    check_dict[("dask_series", "Series")] = check_dask_series
