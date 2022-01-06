# -*- coding: utf-8 -*-
"""Testing utility to compare equality in value for nested objects.

Objects compared can have one of the following valid types:
    types compatible with != comparison
    pd.Series, pd.DataFrame, np.ndarray
    lists, tuples, or dicts of a valid type (recursive)
"""

__author__ = ["fkiraly"]

__all__ = ["deep_equals"]


import numpy as np
import pandas as pd


def deep_equals(x, y, return_msg=False):
    """Test two objects for equality in value.

    Correct if x/y are one of the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)

    Important note:
        this function will return "not equal" if types of x,y are different
        for instant, bool and numpy.bool are *not* considered equal

    Parameters
    ----------
    x : object
    y : object
    return_msg : bool, optional, default=False
        whether to return informative message about what is not equal

    Returns
    -------
    is_equal: bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    msg : str, only returned if return_msg = True
        indication of what is the reason for not being equal
    """

    def ret(is_equal, msg):
        if return_msg:
            if is_equal:
                msg = ""
            return is_equal, msg
        else:
            return is_equal

    if type(x) != type(y):
        return ret(False, "type")

    # we now know all types are the same
    # so now we compare values
    if isinstance(x, pd.Series):
        if x.dtype != y.dtype:
            return ret(False, "dtype")
        # if columns are object, recurse over entries and index
        if x.dtype == "object":
            index_equal = x.index.equals(y.index)
            values_equal, values_msg = deep_equals(
                list(x.values), list(y.values), return_msg=True
            )
            if not index_equal:
                msg = ".index"
            elif not values_equal:
                msg = ".values" + values_msg
            return ret(index_equal and values_equal, msg)
        else:
            return ret(x.equals(y), ".series_equals")
    elif isinstance(x, pd.DataFrame):
        if not x.columns.equals(y.columns):
            return ret(False, ".columns")
        # if columns are equal and at least one is object, recurse over Series
        if sum(x.dtypes == "object") > 0:
            for c in x.columns:
                is_equal, msg = deep_equals(x[c], y[c])
                if not is_equal:
                    return ret(False, f'["{c}"]' + msg)
            return ret(True, "")
        else:
            return ret(x.equals(y), ".df_equals")
    elif isinstance(x, np.ndarray):
        if x.dtype != y.dtype:
            return ret(False, ".dtype")
        return ret(np.array_equal(x, y, equal_nan=True), ".values")
    # recursion through lists, tuples and dicts
    elif isinstance(x, (list, tuple)):
        return ret(*_tuple_equals(x, y, return_msg=True))
    elif isinstance(x, dict):
        return ret(*_dict_equals(x, y, return_msg=True))
    elif x != y:
        return ret(False, " !=")

    return ret(True, "")


def _tuple_equals(x, y, return_msg=False):
    """Test two tuples or lists for equality.

    Correct if tuples/lists contain the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)

    Parameters
    ----------
    x: tuple or list
    y: tuple or list

    Returns
    -------
    bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    """

    def ret(is_equal, msg):
        if return_msg:
            if is_equal:
                msg = ""
            return is_equal, msg
        else:
            return is_equal

    n = len(x)

    if n != len(y):
        return ret(False, ".len")

    # we now know dicts are same length
    for i in range(n):
        xi = x[i]
        yi = y[i]

        # recurse through xi/yi
        is_equal, msg = deep_equals(xi, yi, return_msg=True)
        if not is_equal:
            return ret(False, f"[{i}]" + msg)

    return ret(True, "")


def _dict_equals(x, y, return_msg=False):
    """Test two dicts for equality.

    Correct if dicts contain the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)

    Parameters
    ----------
    x: dict
    y: dict

    Returns
    -------
    bool - True if x and y have equal keys and values
    """

    def ret(is_equal, msg):
        if return_msg:
            if is_equal:
                msg = ""
            return is_equal, msg
        else:
            return is_equal

    xkeys = set(x.keys())
    ykeys = set(y.keys())

    if xkeys != ykeys:
        return ret(False, ".keys")

    # we now know all keys are the same
    for key in xkeys:
        xi = x[key]
        yi = y[key]

        # recurse through xi/yi
        is_equal, msg = deep_equals(xi, yi, return_msg=True)
        if not is_equal:
            return ret(False, f"[{key}]" + msg)

    return ret(True, "")
