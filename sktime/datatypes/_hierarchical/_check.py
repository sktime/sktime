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
        "is_univariate": bool, True iff all series in hier.panel have one variable
        "is_equally_spaced": bool, True iff all series indices are equally spaced
        "is_equal_length": bool, True iff all series in panel are of equal length
        "is_empty": bool, True iff one or more of the series in the panel are empty
        "is_one_series": bool, True iff there is only one series in the hier.panel
        "is_one_panel": bool, True iff there is only one flat panel in the hier.panel
        "has_nans": bool, True iff the panel contains NaN values
        "n_instances": int, number of instances in the hierarchical panel
        "n_panels": int, number of flat panels in the hierarchical panel
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

from sktime.datatypes._series._check import check_pddataframe_series


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


def _ret(valid, msg, metadata, return_metadata):
    if return_metadata:
        return valid, msg, metadata
    else:
        return valid


def check_pdmultiindex_hierarchical(obj, return_metadata=False, var_name="obj"):

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pd.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not isinstance(obj.index, pd.MultiIndex):
        msg = f"{var_name} must have a MultiIndex, found {type(obj.index)}"
        return _ret(False, msg, None, return_metadata)

    # check that columns are unique
    msg = f"{var_name} must have " f"unique column indices, but found {obj.columns}"
    assert obj.columns.is_unique, msg

    # check that there are 3 or more index levels
    nlevels = obj.index.nlevels
    if not nlevels > 2:
        msg = (
            f"{var_name} must have a MultiIndex with 3 or more levels, found {nlevels}"
        )
        return _ret(False, msg, None, return_metadata)

    inst_inds = obj.index.droplevel(-1).unique()
    panel_inds = inst_inds.droplevel(-1).unique()

    check_res = [
        check_pddataframe_series(obj.loc[i], return_metadata=True) for i in inst_inds
    ]
    bad_inds = [i[1] for i in enumerate(inst_inds) if not check_res[i[0]][0]]

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
    metadata["n_panels"] = len(panel_inds)
    metadata["is_one_series"] = len(inst_inds) == 1
    metadata["is_one_panel"] = len(panel_inds) == 1
    metadata["has_nans"] = obj.isna().values.any()
    metadata["is_equal_length"] = _list_all_equal([len(obj.loc[i]) for i in inst_inds])

    return _ret(True, None, metadata, return_metadata)


check_dict[("pd_multiindex_hier", "Hierarchical")] = check_pdmultiindex_hierarchical
