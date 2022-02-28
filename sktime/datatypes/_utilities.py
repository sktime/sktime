# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Eclectic utilities for the datatypes module."""

import numpy as np
import pandas as pd


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

    -------
    index : pandas.Index, index for obj
    """
    if hasattr(obj, "index"):
        return obj.index
    # now we know the object must be an np.ndarray
    return pd.RangeIndex(cutoff, cutoff + obj.shape[0])


def get_cutoff(obj, cutoff=0, return_index=False):
    """Get cutoff = latest time point of time series or time series panel.

    Assumptions on obj are not checked, these should be validated separately.
    Function may return unexpected results without prior validation.

    Parameters
    ----------
    obj : sktime compatible time series data container
        must be of one of the following mtypes:
            pd.Series, pd.DataFrame, np.ndarray, of Series scitype
            pd.multiindex, numpy3D, nested_univ, df-list, of Panel scitype
            pd_multiindex_hier, of Hierarchical scitype
    cutoff : int, optional, default=0
        current cutoff, used to offset index if obj is np.ndarray
    return_index : bool, optional, default=False
        whether a pd.Index object should be returned (True)
            or a pandas compatible index element (False)
        note: return_index=True may set freq attribute of time types to None
            return_index=False will typically preserve freq attribute

    -------
    cutoff_index : pandas compatible index element (if return_index=False)
        pd.Index of length 1 (if return_index=True)
    """
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
        if return_index:
            return pd.RangeIndex(cutoff_ind - 1, cutoff_ind)
        else:
            return cutoff_ind

    if isinstance(obj, pd.Series):
        return obj.index[[-1]] if return_index else obj.index[-1]

    # nested_univ (Panel) or pd.DataFrame(Series)
    if isinstance(obj, pd.DataFrame) and not isinstance(obj.index, pd.MultiIndex):
        objcols = [x for x in obj.columns if obj.dtypes[x] == "object"]
        # pd.DataFrame
        if len(objcols) == 0:
            return obj.index[[-1]] if return_index else obj.index[-1]
        # nested_univ
        else:
            if return_index:
                idxx = [x.index[[-1]] for col in objcols for x in obj[col]]
            else:
                idxx = [x.index[-1] for col in objcols for x in obj[col]]
            return max(idxx)

    # pd-multiindex (Panel) and pd_multiindex_hier (Hierarchical)
    if isinstance(obj, pd.DataFrame) and isinstance(obj.index, pd.MultiIndex):
        idx = obj.index
        nlevels = idx.nlevels
        idx_t = idx.droplevel(list(range(nlevels - 1)))
        return idx_t.sort_values()[[-1]] if return_index else idx_t.sort_values()[-1]

    # df-list (Panel)
    if isinstance(obj, list):
        if return_index:
            idxs = [x.index[[-1]] for x in obj]
        else:
            idxs = [x.index[-1] for x in obj]
        return max(idxs)
