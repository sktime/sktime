"""Machine type checkers for Table scitype.

Exports checkers for Table scitype:

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
        "is_univariate": bool, True iff table has one variable
        "is_empty": bool, True iff table has no variables or no instances
        "has_nans": bool, True iff the panel contains NaN values
        "n_instances": int, number of instances/rows in the table
"""

__author__ = ["fkiraly"]

__all__ = ["check_dict"]

import numpy as np
import pandas as pd

from sktime.datatypes._common import _req, _ret

check_dict = dict()


PRIMITIVE_TYPES = (float, int, str)


def check_pddataframe_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pandas.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = len(obj.columns) < 2
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(index)
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()

    # check that no dtype is object
    if "object" in obj.dtypes.values:
        msg = f"{var_name} should not have column of 'object' dtype"
        return _ret(False, msg, None, return_metadata)

    return _ret(True, None, metadata, return_metadata)


check_dict[("pd_DataFrame_Table", "Table")] = check_pddataframe_table


def check_pdseries_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, pd.Series):
        msg = f"{var_name} must be a pandas.Series, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a pd.Series
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = True
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(index)

    # check that dtype is not object
    if "object" == obj.dtypes:
        msg = f"{var_name} should not be of 'object' dtype"
        return _ret(False, msg, None, return_metadata)

    # check whether index is equally spaced or if there are any nans
    #   compute only if needed
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()

    return _ret(True, None, metadata, return_metadata)


check_dict[("pd_Series_Table", "Table")] = check_pdseries_table


def check_numpy1d_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if len(obj.shape) != 1:
        msg = f"{var_name} must be 1D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 1D np.ndarray
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(obj)
    # 1D numpy arrays are considered univariate
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = True
    # check whether there any nans; compute only if requested
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = pd.isnull(obj).any()

    return _ret(True, None, metadata, return_metadata)


check_dict[("numpy1D", "Table")] = check_numpy1d_table


def check_numpy2d_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if len(obj.shape) != 2:
        msg = f"{var_name} must be 1D or 2D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 2D np.ndarray
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = obj.shape[1] < 2
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = obj.shape[0]
    # check whether there any nans; compute only if requested
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = pd.isnull(obj).any()

    return _ret(True, None, metadata, return_metadata)


check_dict[("numpy2D", "Table")] = check_numpy2d_table


def check_list_of_dict_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, list):
        msg = f"{var_name} must be a list of dict, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not np.all([isinstance(x, dict) for x in obj]):
        msg = (
            f"{var_name} must be a list of dict, but elements at following "
            f"indices are not dict: {np.where(not isinstance(x, dict) for x in obj)}"
        )
        return _ret(False, msg, None, return_metadata)

    for i, d in enumerate(obj):
        for key in d.keys():
            if not isinstance(d[key], PRIMITIVE_TYPES):
                msg = (
                    "all entries must be of primitive type (str, int, float), but "
                    f"found {type(d[key])} at index {i}, key {key}"
                )

    # we now know obj is a list of dict
    # check whether there any nans; compute only if requested
    if _req("is_univariate", return_metadata):
        multivariate_because_one_row = np.any([len(x) > 1 for x in obj])
        if not multivariate_because_one_row:
            all_keys = np.unique([key for d in obj for key in d.keys()])
            multivariate_because_keys_different = len(all_keys) > 1
            multivariate = multivariate_because_keys_different
        else:
            multivariate = multivariate_because_one_row
        metadata["is_univariate"] = not multivariate
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = np.any(
            [pd.isnull(d[key]) for d in obj for key in d.keys()]
        )
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or np.all([len(x) < 1 for x in obj])
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(obj)

    return _ret(True, None, metadata, return_metadata)


check_dict[("list_of_dict", "Table")] = check_list_of_dict_table
