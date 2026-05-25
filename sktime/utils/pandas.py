#!/usr/bin/env python3 -u
"""Utilities for pandas adaptation and version compatibility."""

__author__ = ["fkiraly"]

import re

import pandas as pd
from pandas.tseries.frequencies import to_offset

from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.warnings import _suppress_pd22_warning

__all__ = [
    "df_map",
    "decode_freq_alias",
    "encode_freq_alias",
    "freq_equal",
    "hash_pandas_index",
    "index_sort_values",
    "is_pandas_ge_2_1",
    "is_pandas_ge_3",
    "to_offset_compat",
]

# sktime canonical (legacy) aliases -> pandas 2.1+ / 3 aliases for ``to_offset``
_LEGACY_TO_MODERN_ALIAS = {
    "M": "ME",
    "Q": "QE",
    "A": "YE",
    "Y": "YE",
    "BM": "BME",
    "BQ": "BQE",
    "BA": "BYE",
    "SM": "SME",
}

# pandas 2.1+ / 3 aliases -> sktime canonical (legacy) representation
_MODERN_TO_LEGACY_ALIAS = {
    "ME": "M",
    "QE": "Q",
    "YE": "A",
    "BME": "BM",
    "BQE": "BQ",
    "BYE": "BA",
    "SME": "SM",
}


def is_pandas_ge_2_1():
    """Return whether pandas is version 2.1.0 or newer."""
    return _check_soft_dependencies("pandas>=2.1.0", severity="none")


def is_pandas_ge_3():
    """Return whether pandas is version 3.0.0 or newer."""
    return _check_soft_dependencies("pandas>=3.0.0", severity="none")


def _split_freq_alias(freqstr):
    """Split a frequency string into numeric multiplier and base alias."""
    if not isinstance(freqstr, str):
        return "", freqstr
    match = re.fullmatch(r"(\d*)(.+)", freqstr)
    if match is None:
        return "", freqstr
    mult, base = match.groups()
    return mult, base


def encode_freq_alias(freqstr):
    """Encode sktime/legacy frequency aliases for the installed pandas version.

    On pandas >= 2.1, legacy aliases such as ``M`` are converted to modern
    aliases such as ``ME`` before calling ``to_offset``. On older pandas, the
    string is returned unchanged.

    Parameters
    ----------
    freqstr : str or None

    Returns
    -------
    str or None
    """
    if freqstr is None or not is_pandas_ge_2_1():
        return freqstr

    mult, base = _split_freq_alias(freqstr)
    base_enc = _LEGACY_TO_MODERN_ALIAS.get(base, base)
    return f"{mult}{base_enc}" if mult else base_enc


def decode_freq_alias(freqstr):
    """Decode pandas 2.1+ frequency aliases to sktime canonical aliases.

    On pandas < 2.1, modern aliases such as ``ME`` are not used by pandas and
    the string is returned unchanged.

    Parameters
    ----------
    freqstr : str or None

    Returns
    -------
    str or None
    """
    if freqstr is None or _check_soft_dependencies("pandas<2.1.0", severity="none"):
        return freqstr

    mult, base = _split_freq_alias(freqstr)
    base_dec = _MODERN_TO_LEGACY_ALIAS.get(base, base)
    return f"{mult}{base_dec}" if mult else base_dec


def _freq_for_to_offset(freqstr):
    """Normalize a frequency string for ``to_offset`` on the installed pandas.

    On pandas >= 2.1, legacy aliases are encoded to modern aliases.
    On pandas < 2.1, modern aliases are mapped to legacy aliases.
    """
    if freqstr is None:
        return None

    mult, base = _split_freq_alias(freqstr)
    if is_pandas_ge_2_1():
        base_norm = _LEGACY_TO_MODERN_ALIAS.get(base, base)
    else:
        base_norm = _MODERN_TO_LEGACY_ALIAS.get(base, base)
    return f"{mult}{base_norm}" if mult else base_norm


def to_offset_compat(freq):
    """Return a pandas offset, using version-appropriate frequency aliases.

    Parameters
    ----------
    freq : str, pandas offset, or None

    Returns
    -------
    pandas offset or None
    """
    if isinstance(freq, pd.offsets.BaseOffset):
        return freq
    if freq is None:
        with _suppress_pd22_warning():
            return to_offset(freq)
    # freq_enc = encode_freq_alias(freq) if isinstance(freq, str) else freq
    freq_enc = _freq_for_to_offset(freq) if isinstance(freq, str) else freq
    with _suppress_pd22_warning():
        return to_offset(freq_enc)


def freq_equal(freq_a, freq_b):
    """Compare two frequencies, accounting for pandas alias differences.

    Parameters
    ----------
    freq_a, freq_b : str, pandas offset, or None

    Returns
    -------
    bool
    """
    if freq_a is None and freq_b is None:
        return True
    if freq_a is None or freq_b is None:
        return False

    offset_a = to_offset_compat(freq_a)
    offset_b = to_offset_compat(freq_b)
    with _suppress_pd22_warning():
        return offset_a == offset_b


def index_sort_values(index):
    """Sort a pandas Index in a version-compatible way.

    Parameters
    ----------
    index : pd.Index

    Returns
    -------
    pd.Index
    """
    return index.sort_values()


def hash_pandas_index(index):
    """Return a stable hash for a pandas Index (for caching).

    Parameters
    ----------
    index : pd.Index

    Returns
    -------
    int
    """
    try:
        return int(pd.util.hash_pandas_object(index).sum())
    except (TypeError, AttributeError, ValueError):
        pass

    if hasattr(pd.util, "hash_tuples"):
        try:
            return int(pd.util.hash_tuples(index).sum())
        except (TypeError, AttributeError, ValueError):
            pass

    return hash(tuple(index.to_numpy().tolist()))


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
