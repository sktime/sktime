#!/usr/bin/env python3 -u
"""Utilities for pandas adapbation."""

__author__ = ["fkiraly"]


def df_map(x):
    """Access map or applymap, of DataFrame.

    In pandas 2.1.0, applymap was deprecated in favor of the newly introduced map.
    To ensure compatibility with older versions, we use map if available,
    otherwise applymap.

    Parameters
    ----------
    x : assumed pd.DataFrame

    Returns
    -------
    x.map, if available, otherwise x.applymap
        Note: returns method itself, not result of method call
    """
    if hasattr(x, "map"):
        return x.map
    else:
        return x.applymap
