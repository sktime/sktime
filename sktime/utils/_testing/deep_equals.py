# -*- coding: utf-8 -*-
"""
Testing utility to compare equality in value for nested objects

Objects compared can have one of the following valid types:
    types compatible with != comparison
    pd.Series, pd.DataFrame, np.ndarray
    lists, tuples, or dicts of a valid type (recursive)
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd


def deep_equals(x, y):
    """Tests two objects for equality in value

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
    if type(x) in [pd.DataFrame, pd.Series]:
        if not x.equals(y):
            return False
    elif type(x) is np.ndarray:
        if x.dtype != y.dtype:
            return False
        if not np.array_equal(x, y, equal_nan=True):
            return False
    # recursion through lists, tuples and dicts
    elif type(x) in [list, tuple]:
        return tuple_equals(x, y)
    elif type(x) is dict:
        return dict_equals(x, y)
    elif x != y:
        return False

    return True


def tuple_equals(x, y):
    """Tests two tuples or lists for equality.

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


def dict_equals(x, y):
    """Tests two dicts for equality.

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
