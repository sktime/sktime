#!/usr/bin/env python3 -u
"""Multiindex formatting related utilities."""

__author__ = ["fkiraly", "ksharma6"]
__all__ = []

import numpy as np
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


def apply_split(y, iloc_ix):
    """Generate iloc indices to split according to first MultiIndex level.

    Applies iloc_ix to the leftmost (0-th) level of a MultiIndex y.

    Parameters
    ----------
    y : pd.MultiIndex or pd.Index
        Index to split, coerced to MultiIndex if not already
    iloc_ix: 1D np.ndarray of integer
        iloc indices to apply to the first level of y

    Returns
    -------
    y_iloc : ndarray
        iloc indices for y after applying iloc_ix to the first level

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.utils.multiindex import apply_split
    >>> y = pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    >>> iloc_ix = np.array([1, 0])
    >>> apply_split(y, iloc_ix)  # doctest: +SKIP
    array([2, 3, 0, 1])
    """
    if not isinstance(y, pd.MultiIndex):
        zeros = [0] * len(y)
        y = pd.MultiIndex.from_arrays([zeros, y])

    if not isinstance(iloc_ix, np.ndarray):
        iloc_ix = np.array(iloc_ix)

    inst_ix = y.droplevel(-1).unique()
    iloc_ixer = pd.DataFrame(pd.RangeIndex(len(y)), index=y)

    y_loc = inst_ix[np.array(iloc_ix)]
    y_np = [iloc_ixer.loc[x].to_numpy().flatten() for x in y_loc]
    y_iloc = np.concatenate(y_np)
    return y_iloc


def apply_method_per_series(y, method_name, *args, **kwargs):
    """
    Apply a method to each series in a multiindex pandas object.

    Parameters
    ----------
    y : pd.DataFrame or pd.Series
        Data to apply method to
    method_name : str
        Name of method to apply
    args : list
        Positional arguments to pass to method
    kwargs : dict
        Keyword arguments to pass to method

    Returns
    -------
    pd.DataFrame or pd.Series
        Data after applying method to each series
    """
    if y.index.nlevels == 1:
        # Apply method directly
        return getattr(y, method_name)(*args, **kwargs)

    series_idx_tuples = y.index.droplevel(-1).unique().to_list()
    series_list = []
    for group_keys in series_idx_tuples:
        y_series = y.loc[group_keys]
        y_series = getattr(y_series, method_name)(*args, **kwargs)
        # Add multiindex
        y_series.index = pd.MultiIndex.from_tuples(
            [(*group_keys, idx) for idx in y_series.index], names=y.index.names
        )
        series_list.append(y_series)
    y = pd.concat(series_list).sort_index()
    return y


def is_hierarchical(multiindex: pd.Index, raise_if_false=False) -> bool:
    """Determine if a pandas MultiIndex is strictly hierarchical.

    Strictly hierarchical means that each child-level value corresponds to one and
    only one parent-level value.

    If a regular index is passed, it is considered hierarchical (single level).

    Parameters
    ----------
    multiindex (pd.Index or pd.DataFrame): The MultiIndex (or DataFrame) to check.
    raise_if_false (bool, optional): If set to True, an exception is raised when
        the index is not hierarchical. Default is False.

    Returns
    -------
    bool: True if the MultiIndex is hierarchical, False otherwise.

    Raises
    ------
    Exception: If `raise_if_false` is True and the MultiIndex is not hierarchical,
               an exception is raised with details about the issue.
    """
    # Handle the case where a DataFrame is passed
    if isinstance(multiindex, pd.DataFrame):
        multiindex = multiindex.index

    # If the index is not a MultiIndex, return True as it is "inherently hierarchical"
    if not isinstance(multiindex, pd.MultiIndex):
        return True

    # Determine if the MultiIndex is truly hierarchical
    for level in range(len(multiindex.levels) - 1):
        parent = multiindex.get_level_values(level)
        child = multiindex.get_level_values(level + 1)
        mapping = pd.DataFrame(
            {f"{level}": parent, f"{level + 1}": child}
        ).drop_duplicates()

        # Check if any child value appears under multiple parent values
        if mapping.duplicated(subset=f"{level + 1}").any():
            if raise_if_false:
                dups = mapping[
                    mapping.duplicated(subset=f"{level + 1}", keep=False)
                ].to_string()
                msg = f"Duplicate child values found for level: {level + 1}\n{dups}"
                raise Exception(msg)
            else:
                return False
    return True
