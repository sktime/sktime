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
        "n_features": int, number of variables in series
        "feature_names": list of int or object, names of variables in series
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

from sktime.datatypes._common import _req
from sktime.datatypes._common import _ret as ret
from sktime.datatypes._dtypekind import _get_feature_kind, _get_series_dtypekind
from sktime.utils.dependencies import _check_soft_dependencies
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

    # check to delineate from nested_univ mtype (Panel)
    # pd.DataFrame mtype allows object dtype,
    # but if we allow object dtype with pd.Series entries,
    # the mtype becomes ambiguous, i.e., non-delineable from nested_univ
    if np.prod(obj.shape) > 0 and isinstance(obj.iloc[0, 0], (pd.Series, pd.DataFrame)):
        msg = f"{var_name} cannot contain nested pd.Series or pd.DataFrame"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = len(obj.columns) < 2
    if _req("n_features", return_metadata):
        metadata["n_features"] = len(obj.columns)
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns.to_list()
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "pd.DataFrame")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_series_dtypekind(obj, "pd.DataFrame")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

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
    if _req("n_features", return_metadata):
        metadata["n_features"] = 1
    if _req("feature_names", return_metadata):
        if not hasattr(obj, "name") or obj.name is None:
            metadata["feature_names"] = [0]
        else:
            metadata["feature_names"] = [obj.name]
    if _req("dtypekind_dfip", return_metadata):
        metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "pd.Series")
    if _req("feature_kind", return_metadata):
        dtype_kind = _get_series_dtypekind(obj, "pd.Series")
        metadata["feature_kind"] = _get_feature_kind(dtype_kind)

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
        if _req("n_features", return_metadata):
            metadata["n_features"] = obj.shape[1]
        if _req("feature_names", return_metadata):
            metadata["feature_names"] = list(range(obj.shape[1]))
        if _req("dtypekind_dfip", return_metadata):
            metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "numpy")
        if _req("feature_kind", return_metadata):
            dtype_kind = _get_series_dtypekind(obj, "numpy")
            metadata["feature_kind"] = _get_feature_kind(dtype_kind)
    elif len(obj.shape) == 1:
        # we now know obj is a 1D np.ndarray
        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(obj) < 1
        if _req("is_univariate", return_metadata):
            metadata["is_univariate"] = True
        if _req("n_features", return_metadata):
            metadata["n_features"] = 1
        if _req("feature_names", return_metadata):
            metadata["feature_names"] = [0]
        if _req("dtypekind_dfip", return_metadata):
            metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "numpy")
        if _req("feature_kind", return_metadata):
            dtype_kind = _get_series_dtypekind(obj, "numpy")
            metadata["feature_kind"] = _get_feature_kind(dtype_kind)
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
        if len(obj.dims) == 1:
            if _req("n_features", return_metadata):
                metadata["n_features"] = 1
            if _req("feature_names", return_metadata):
                metadata["feature_names"] = [0]
        else:
            if _req("n_features", return_metadata):
                metadata["n_features"] = len(obj[obj.dims[1]])
            if _req("feature_names", return_metadata):
                metadata["feature_names"] = obj.indexes[obj.dims[1]].to_list()

        if _req("dtypekind_dfip", return_metadata):
            metadata["dtypekind_dfip"] = _get_series_dtypekind(obj, "xarray")
        if _req("feature_kind", return_metadata):
            dtype_kind = _get_series_dtypekind(obj, "xarray")
            metadata["feature_kind"] = _get_feature_kind(dtype_kind)

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

if _check_soft_dependencies("gluonts", severity="none"):

    def check_gluonTS_listDataset_series(obj, return_metadata=False, var_name="obj"):
        metadata = dict()

        if (
            not isinstance(obj, list)
            or not isinstance(obj[0], dict)
            or "target" not in obj[0]
            or len(obj[0]["target"]) > 1
        ):
            msg = f"{var_name} must be a gluonts.ListDataset, found {type(obj)}"
            return ret(False, msg, None, return_metadata)

        # Check if there are no time series in the ListDataset
        if _req("is_empty", return_metadata):
            metadata["is_empty"] = len(obj) < 1

        if _req("is_univariate", return_metadata):
            # Check first if the ListDataset is empty
            if len(obj) < 1:
                metadata["is_univariate"] = True

            # Check the first time-series for total features
            else:
                metadata["is_univariate"] = obj[0]["target"].shape[1] == 1

        if _req("n_features", return_metadata):
            # Check first if the ListDataset is empty
            if len(obj) < 1:
                metadata["n_features"] = 0

            else:
                metadata["n_features"] = obj[0]["target"].shape[1]

        if _req("n_instances", return_metadata):
            metadata["n_instances"] = 1

        if _req("feature_names", return_metadata):
            # Check first if the ListDataset is empty
            if len(obj) < 1:
                metadata["feature_names"] = []

            else:
                metadata["feature_names"] = [
                    f"value_{i}" for i in range(obj[0]["target"].shape[1])
                ]

        for series in obj:
            # check that no dtype is object
            if series["target"].dtype == "object":
                msg = f"{var_name} should not have column of 'object' dtype"
                return ret(False, msg, None, return_metadata)

        # Check if a valid Frequency is set
        if FREQ_SET_CHECK and len(obj) >= 1:
            if obj[0].freq is None:
                msg = f"{var_name} has no freq attribute set."
                return ret(False, msg, None, return_metadata)

        # For a GluonTS ListDataset, only a start date and frequency is set
        # so everything should thus be equally spaced
        if _req("is_equally_spaced", return_metadata):
            metadata["is_equally_spaced"] = True

        if _req("has_nans", return_metadata):
            for series in obj:
                metadata["has_nans"] = pd.isnull(series["target"]).any()

                # Break out if at least 1 time series has NaN values
                if metadata["has_nans"]:
                    break

        return ret(True, None, metadata, return_metadata)

    check_dict[("gluonts_ListDataset_series", "Series")] = (
        check_gluonTS_listDataset_series
    )
