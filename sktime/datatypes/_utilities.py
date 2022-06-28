# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Eclectic utilities for the datatypes module."""

import numpy as np
import pandas as pd


def _get_index(x):
    if hasattr(x, "index"):
        return x.index
    else:
        # select last dimension for time index
        return pd.RangeIndex(x.shape[-1])


def get_time_index(X):
    """Get index of time series data, helper function.

    Parameters
    ----------
    X : pd.DataFrame / pd.Series / np.ndarray
    in one of the following sktime mtype specifications for Series, Panel, Hierarchical:
    pd.DataFrame, pd.Series, np.ndarray, pd-multiindex, nested_univ, pd_multiindex_hier
    assumes all time series have equal length and equal index set
    will *not* work for list-of-df, pd-wide, pd-long

    Returns
    -------
    time_index : pandas Index
        Index of time series
    """
    # assumes that all samples share the same the time index, only looks at
    # first row
    if isinstance(X, (pd.DataFrame, pd.Series)):
        # pd-multiindex or pd_multiindex_hier
        if isinstance(X.index, pd.MultiIndex):
            first_inst = X.index.to_flat_index()[0][:-1]
            return X.loc[first_inst].index
        # nested_univ
        elif isinstance(X, pd.DataFrame) and isinstance(X.iloc[0, 0], pd.DataFrame):
            return _get_index(X.iloc[0, 0])
        # pd.Series or pd.DataFrame
        else:
            return X.index
    # numpy3D and np.ndarray
    elif isinstance(X, np.ndarray):
        return _get_index(X)
    else:
        raise ValueError(
            f"X must be a pandas DataFrame or Series, but found: {type(X)}"
        )


def get_index_for_series(obj, cutoff=0):
    """Get pandas index for a Series object.

    Returns index even for numpy array, in that case a RangeIndex.

    Assumptions on obj are not checked, these should be validated separately.
    Function may return unexpected results without prior validation.

    Parameters
    ----------
    obj : sktime data container
        must be of one of the following mtypes:
            pd.Series, pd.DataFrame, np.ndarray, of Series scitype
    cutoff : int, or pd.datetime, optional, default=0
        current cutoff, used to offset index if obj is np.ndarray

    Returns
    -------
    index : pandas.Index, index for obj
    """
    if hasattr(obj, "index"):
        return obj.index
    # now we know the object must be an np.ndarray
    return pd.RangeIndex(cutoff, cutoff + obj.shape[0])


GET_CUTOFF_SUPPORTED_MTYPES = [
    "pd.DataFrame",
    "pd.Series",
    "np.ndarray",
    "pd-multiindex",
    "numpy3D",
    "nested_univ",
    "df-list",
    "pd_multiindex_hier",
    "np.ndarray",
    "numpy3D",
]


def get_cutoff(
    obj,
    cutoff=0,
    return_index=False,
    reverse_order=False,
    check_input=False,
    convert_input=True,
):
    """Get cutoff = latest time point of time series or time series panel.

    Assumptions on obj are not checked, these should be validated separately.
    Function may return unexpected results without prior validation.

    Parameters
    ----------
    obj : sktime compatible time series data container
        must be of Series, Panel, or Hierarchical scitype
        all mtypes are supported via conversion to internally supported types
        to avoid conversions, pass data in one of GET_CUTOFF_SUPPORTED_MTYPES
    cutoff : int, optional, default=0
        current cutoff, used to offset index if obj is np.ndarray
    return_index : bool, optional, default=False
        whether a pd.Index object should be returned (True)
            or a pandas compatible index element (False)
        note: return_index=True may set freq attribute of time types to None
            return_index=False will typically preserve freq attribute
    reverse_order : bool, optional, default=False
        if False, returns largest time index. If True, returns smallest time index
    check_input : bool, optional, default=False
        whether to check input for validity, i.e., is it one of the scitypes
    convert_input : bool, optional, default=True
        whether to convert the input (True), or skip conversion (False)
        if skipped, function assumes that inputs are one of GET_CUTOFF_SUPPORTED_MTYPES

    Returns
    -------
    cutoff_index : pandas compatible index element (if return_index=False)
        pd.Index of length 1 (if return_index=True)

    Raises
    ------
    ValueError, TypeError, if check_input or convert_input are True
        exceptions from check or conversion failure, in check_is_scitype, convert_to
    """
    from sktime.datatypes import check_is_scitype, convert_to

    # deal with VectorizedDF
    if hasattr(obj, "X"):
        obj = obj.X

    if check_input:
        valid = check_is_scitype(obj, scitype=["Series", "Panel", "Hierarchical"])
        if not valid:
            raise ValueError("obj must be of Series, Panel, or Hierarchical scitype")

    if convert_input:
        obj = convert_to(obj, GET_CUTOFF_SUPPORTED_MTYPES)

    if cutoff is None:
        cutoff = 0

    if len(obj) == 0:
        return cutoff

    # numpy3D (Panel) or np.npdarray (Series)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3:
            cutoff_ind = obj.shape[-1] + cutoff
        if obj.ndim < 3 and obj.ndim > 0:
            cutoff_ind = obj.shape[0] + cutoff
        if reverse_order:
            cutoff_ind = 0
        if return_index:
            return pd.RangeIndex(cutoff_ind - 1, cutoff_ind)
        else:
            return cutoff_ind

    # define "first" or "last" index depending on which is desired
    if reverse_order:
        ix = 0
        agg = min
    else:
        ix = -1
        agg = max

    if isinstance(obj, pd.Series):
        return obj.index[[ix]] if return_index else obj.index[ix]

    # nested_univ (Panel) or pd.DataFrame(Series)
    if isinstance(obj, pd.DataFrame) and not isinstance(obj.index, pd.MultiIndex):
        objcols = [x for x in obj.columns if obj.dtypes[x] == "object"]
        # pd.DataFrame
        if len(objcols) == 0:
            return obj.index[[ix]] if return_index else obj.index[ix]
        # nested_univ
        else:
            if return_index:
                idxx = [x.index[[ix]] for col in objcols for x in obj[col]]
            else:
                idxx = [x.index[ix] for col in objcols for x in obj[col]]
            return max(idxx)

    # pd-multiindex (Panel) and pd_multiindex_hier (Hierarchical)
    if isinstance(obj, pd.DataFrame) and isinstance(obj.index, pd.MultiIndex):
        idx = obj.index
        series_idx = [obj.loc[x].index.get_level_values(-1) for x in idx.droplevel(-1)]
        if return_index:
            cutoffs = [x[[-1]] for x in series_idx]
        else:
            cutoffs = [x[-1] for x in series_idx]
        return agg(cutoffs)

    # df-list (Panel)
    if isinstance(obj, list):
        if return_index:
            idxs = [x.index[[ix]] for x in obj]
        else:
            idxs = [x.index[ix] for x in obj]
        return agg(idxs)


def update_data(X, X_new=None):
    """Update time series container with another one.

    Parameters
    ----------
    X : sktime data container, in one of the following mtype formats
        pd.DataFrame, pd.Series, np.ndarray, pd-multiindex, numpy3D,
        pd_multiindex_hier
    X_new : None, or sktime data container, should be same mtype as X,
        or convert to same format when converting to format list via convert_to

    Returns
    -------
    X updated with X_new, with rows/indices in X_new added
        entries in X_new overwrite X if at same index
        numpy based containers will always be interpreted as having new row index
    """
    from sktime.datatypes._convert import convert_to
    from sktime.datatypes._vectorize import VectorizedDF

    # we only need to modify _X if X is not None
    if X_new is None:
        return X

    # if X is vectorized, unwrap it first
    if isinstance(X, VectorizedDF):
        X = X.X
    # we want to ensure that X is either numpy (1D, 2D, 3D)
    # or in one of the long pandas formats
    X = convert_to(
        X,
        to_type=[
            "pd.DataFrame",
            "pd.Series",
            "np.ndarray",
            "pd-multiindex",
            "numpy3D",
            "pd_multiindex_hier",
        ],
    )
    # update X with the new rows in X_new
    #  if X is np.ndarray, we assume all rows are new
    if isinstance(X, np.ndarray):
        # if 1D or 2D, axis 0 is "time"
        if X_new.ndim in [1, 2]:
            return np.concatenate(X, X_new, axis=0)
        # if 3D, axis 2 is "time"
        elif X_new.ndim == 3:
            return np.concatenate(X, X_new, axis=2)
    #  if y is pandas, we use combine_first to update
    elif isinstance(X_new, (pd.Series, pd.DataFrame)) and len(X_new) > 0:
        return X_new.combine_first(X)


GET_WINDOW_SUPPORTED_MTYPES = [
    "pd.DataFrame",
    "pd-multiindex",
    "pd_multiindex_hier",
    "np.ndarray",
    "numpy3D",
]


def get_window(obj, window_length=None, lag=None):
    """Slice obj to the time index window with given length and lag.

    Returns time series or time series panel with time indices
        strictly greater than cutoff - lag - window_length, and
        equal or less than cutoff - lag.
    Cutoff if of obj, as determined by get_cutoff.

    Parameters
    ----------
    obj : sktime compatible time series data container or None
        if not None, must be of Series, Panel, or Hierarchical scitype
        all mtypes are supported via conversion to internally supported types
        to avoid conversions, pass data in one of GET_WINDOW_SUPPORTED_MTYPES
    window_length : int or timedelta, optional, default=-inf
        must be int if obj is int indexed, timedelta if datetime indexed
        length of the window to slice to. Default = window of infinite size
    lag : int, timedelta, or None optional, default = None (zero of correct type)
        lag of the latest time in the window, with respect to cutoff of obj
        if None, is internally replaced by a zero of type compatible with obj index
        must be int if obj is int indexed or not pandas based
        must be timedelta if obj is pandas based and datetime indexed

    Returns
    -------
    obj sub-set to time indices in the semi-open interval
        (cutoff - window_length - lag, cutoff - lag)
        None if obj was None
    """
    from sktime.datatypes import check_is_scitype, convert_to

    if obj is None or (window_length is None and lag is None):
        return obj

    valid, _, metadata = check_is_scitype(
        obj, scitype=["Series", "Panel", "Hierarchical"], return_metadata=True
    )
    if not valid:
        raise ValueError("obj must be of Series, Panel, or Hierarchical scitype")
    obj_in_mtype = metadata["mtype"]

    obj = convert_to(obj, GET_WINDOW_SUPPORTED_MTYPES)

    # numpy3D (Panel) or np.npdarray (Series)
    if isinstance(obj, np.ndarray):
        # if 2D or 3D, we need to subset by last, not first dimension
        # if 1D, we need to subset by first dimension
        # to achieve that effect, we swap first and last in case of 2D, 3D
        # and always subset on first dimension
        if obj.ndim > 1:
            obj = obj.swapaxes(1, -1)
        obj_len = len(obj)
        if lag is None:
            lag = 0
        if window_length is None:
            window_length = obj_len
        window_start = max(-window_length - lag, -obj_len)
        window_end = max(-lag, -obj_len)
        # we need to swap first and last dimension back before returning, if done above
        if window_end == 0:
            obj_subset = obj[window_start:]
        else:
            obj_subset = obj[window_start:window_end]
        if obj.ndim > 1:
            obj_subset = obj_subset.swapaxes(1, -1)
        return obj_subset

    # pd.DataFrame(Series), pd-multiindex (Panel) and pd_multiindex_hier (Hierarchical)
    if isinstance(obj, pd.DataFrame):
        cutoff = get_cutoff(obj)

        if not isinstance(obj.index, pd.MultiIndex):
            time_indices = obj.index
        else:
            time_indices = obj.index.get_level_values(-1)

        if lag is None:
            win_end_incl = cutoff
            win_select = time_indices <= win_end_incl
            if window_length is not None:
                win_start_excl = cutoff - window_length
                win_select = win_select & (time_indices > win_start_excl)
        else:
            win_end_incl = cutoff - lag
            win_select = time_indices <= win_end_incl
            if window_length is not None:
                win_start_excl = cutoff - window_length - lag
                win_select = win_select & (time_indices > win_start_excl)

        obj_subset = obj.iloc[win_select]

        return convert_to(obj_subset, obj_in_mtype)

    raise ValueError(
        "bug in get_window, unreachable condition, ifs should be exhaustive"
    )


def get_slice(obj, start=None, end=None):
    """Slice obj with start (inclusive) and end (exclusive) indices.

    Returns time series or time series panel with time indices
        strictly greater and equal to start index and less than
        end index.

    Parameters
    ----------
    obj : sktime compatible time series data container or None
        if not None, must be of Series, Panel, or Hierarchical scitype
        all mtypes are supported via conversion to internally supported types
        to avoid conversions, pass data in one of GET_WINDOW_SUPPORTED_MTYPES
    start : int or timestamp, optional, default = None
        must be int if obj is int indexed, timestamp if datetime indexed
        Inclusive start of slice. Default = None.
        If None, then no slice at the start
    end : int or timestamp, optional, default = None
        must be int if obj is int indexed, timestamp if datetime indexed
        Exclusive end of slice. Default = None
        If None, then no slice at the end
    Returns
    -------
    obj sub-set sliced for `start` (inclusive) and `end` (exclusive) indices
        None if obj was None
    """
    from sktime.datatypes import check_is_scitype, convert_to

    if (start is None and end is None) or obj is None:
        return obj

    valid, _, metadata = check_is_scitype(
        obj, scitype=["Series", "Panel", "Hierarchical"], return_metadata=True
    )
    if not valid:
        raise ValueError("obj must be of Series, Panel, or Hierarchical scitype")
    obj_in_mtype = metadata["mtype"]

    obj = convert_to(obj, GET_WINDOW_SUPPORTED_MTYPES)

    # numpy3D (Panel) or np.npdarray (Series)
    # Assumes the index is integer so will be exclusive by default
    if isinstance(obj, np.ndarray):
        # if 2D or 3D, we need to subset by last, not first dimension
        # if 1D, we need to subset by first dimension
        # to achieve that effect, we swap first and last in case of 2D, 3D
        # and always subset on first dimension
        if obj.ndim > 1:
            obj = obj.swapaxes(1, -1)
        # subsetting
        if start and end:
            obj_subset = obj[start:end]
        elif end:
            obj_subset = obj[:end]
        else:
            obj_subset = obj[start:]
        # we need to swap first and last dimension back before returning, if done above
        if obj.ndim > 1:
            obj_subset = obj_subset.swapaxes(1, -1)
        return obj_subset

    # pd.DataFrame(Series), pd-multiindex (Panel) and pd_multiindex_hier (Hierarchical)
    # Assumes the index is pd.Timestamp or pd.Period and ensures the end is
    # exclusive with slice_select
    if isinstance(obj, pd.DataFrame):
        if not isinstance(obj.index, pd.MultiIndex):
            time_indices = obj.index
        else:
            time_indices = obj.index.get_level_values(-1)

        if start and end:
            slice_select = (time_indices >= start) & (time_indices < end)
        elif end:
            slice_select = time_indices < end
        elif start:
            slice_select = time_indices >= start

        obj_subset = obj.iloc[slice_select]
        return convert_to(obj_subset, obj_in_mtype)

    raise ValueError(
        "bug in get_slice, unreachable condition, ifs should be exhaustive"
    )
