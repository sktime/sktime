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


def deep_equals(x, y):
    """Test two objects for equality in value.

    Correct if x/y are one of the following valid types:
        types compatible with != comparison
        pd.Series, pd.DataFrame, np.ndarray
        lists, tuples, or dicts of a valid type (recursive)

    Parameters
    ----------
    x: object
    y: object

    Returns
    -------
    bool - True if x and y are equal in value
        x and y do not need to be equal in reference
    """
    if type(x) != type(y):
        return False

    # we now know all types are the same
    # so now we compare values
    if isinstance(x, pd.Series):
        if x.dtype != y.dtype:
            return False
        # if columns are object, recurse over entries and index
        if x.dtype == "object":
            index_equal = x.index.equals(y.index)
            return index_equal and deep_equals(list(x.values), list(y.values))
        else:
            return x.equals(y)
    elif isinstance(x, pd.DataFrame):
        if not x.columns.equals(y.columns):
            return False
        # if columns are equal and at least one is object, recurse over Series
        if sum(x.dtypes == "object") > 0:
            return np.all([deep_equals(x[c], y[c]) for c in x.columns])
        else:
            return x.equals(y)
    elif isinstance(x, np.ndarray):
        if x.dtype != y.dtype:
            return False
        return np.array_equal(x, y, equal_nan=True)
    # recursion through lists, tuples and dicts
    elif isinstance(x, (list, tuple)):
        return _tuple_equals(x, y)
    elif isinstance(x, dict):
        return _dict_equals(x, y)
    elif x != y:
        return False

    return True


def _tuple_equals(x, y):
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
    n = len(x)

    if n != len(y):
        return False

    # we now know dicts are same length
    for i in range(n):
        xi = x[i]
        yi = y[i]

        # recurse through xi/yi
        if not deep_equals(xi, yi):
            return False

    return True


def _dict_equals(x, y):
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
    xkeys = set(x.keys())
    ykeys = set(y.keys())

    if xkeys != ykeys:
        return False

    # we now know all keys are the same
    for key in xkeys:
        xi = x[key]
        yi = y[key]

        # recurse through xi/yi
        if not deep_equals(xi, yi):
            return False

    return True
