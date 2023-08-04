#!/usr/bin/env python3 -u
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


def rename_multiindex(idx, feature_names_out, idx_name="index"):
    """Rename (column) multiindex by common sktime renaming convention.

    Parameters
    ----------
    idx: pandas.MultiIndex

    feature_names_out : str, one of "flat" (default), "multiindex", "original", "auto"
        determines the output index converted to
        "flat": index is flattened by application of ``flatten_multiindex``
        "multiindex": index is returned unchanged
        "original": last index level is returned (without type casting)
            if this results in non-unique index, ValueError exception is raised
        "auto": as "original" (cast to str) for any unique columns under "original",
            column names as "flat" otherwise; names are always cast to str
    """
    if feature_names_out == "multiindex":
        return idx
    elif feature_names_out == "flat":
        return flatten_multiindex(idx)
    elif feature_names_out == "original":
        if idx.get_level_values(-1).is_unique:
            return idx.get_level_values(-1)
        else:
            raise ValueError(
                'Error, resulting index names when using "original" naming '
                f"for {idx_name} contains non-unique elements."
            )
    elif feature_names_out == "auto":
        original = idx.get_level_values(-1)
        idx_out = original.copy().values.astype("str")
        if original.is_unique:
            return pd.Index(idx_out)

        flat = flatten_multiindex(idx)
        duplicated = original.duplicated(keep=False)

        idx_out[duplicated] = flat[duplicated]
        return pd.Index(idx_out)
    else:
        raise ValueError(
            "invalid value for feature_names_out in rename_multiindex, "
            'must be one of "flat", "multiindex", "original", "auto", '
            f"but found {feature_names_out}"
        )
