#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Multiindex formatting related utilities."""

__author__ = ["fkiraly"]
__all__ = []

import pandas as pd


def underscore_join(iterable):
    """Create flattened column names from multiindex tuple.

    Parameters
    ----------
    iterable : an iterable

    Returns
    -------
    str, for an iterable (x1, x2, ..., xn), returns the string
        str(x1) + "__" + str(x2) + "__" + str(x3) + ... + "__" + str(xn)
    """
    iterable_as_str = [str(x) for x in iterable]
    return "__".join(iterable_as_str)


def flatten_multiindex(idx):
    """Flatten a multiindex.

    Parameters
    ----------
    idx: pandas.MultiIndex

    Returns
    -------
    pandas.Index with str elements
        i-th element of return is underscore_join of i-th element of `idx`
    """
    return pd.Index([underscore_join(x) for x in idx])
