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
        "is_univariate": bool, True iff all series in hier.panel have one variable
        "is_equally_spaced": bool, True iff all series indices are equally spaced
        "is_equal_length": bool, True iff all series in panel are of equal length
        "is_empty": bool, True iff one or more of the series in the panel are empty
        "is_one_series": bool, True iff there is only one series in the hier.panel
        "is_one_panel": bool, True iff there is only one flat panel in the hier.panel
        "has_nans": bool, True iff the panel contains NaN values
        "n_instances": int, number of instances in the hierarchical panel
        "n_panels": int, number of flat panels in the hierarchical panel
        "n_features": int, number of variables in series
        "feature_names": list of int or object, names of variables in series
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np

from sktime.datatypes._panel._check import check_pdmultiindex_panel
from sktime.utils.dependencies import _check_soft_dependencies


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


def check_pdmultiindex_hierarchical(obj, return_metadata=False, var_name="obj"):
    ret = check_pdmultiindex_panel(
        obj, return_metadata=return_metadata, var_name=var_name, panel=False
    )

    return ret


check_dict[("pd_multiindex_hier", "Hierarchical")] = check_pdmultiindex_hierarchical


if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import check_dask_frame

    def check_dask_hierarchical(obj, return_metadata=False, var_name="obj"):
        return check_dask_frame(
            obj=obj,
            return_metadata=return_metadata,
            var_name=var_name,
            scitype="Hierarchical",
        )

    check_dict[("dask_hierarchical", "Hierarchical")] = check_dask_hierarchical
