"""Utilities for Hyper-Trees forecasters."""

import re

import pandas as pd

_PANDAS_FREQUENCY_ALIASES = {
    "A": "YE",
    "Y": "YE",
    "Q": "QE",
    "BQ": "BQE",
    "SM": "SME",
    "BM": "BME",
    "CBM": "CBME",
    "H": "h",
    "T": "min",
    "S": "s",
    "L": "ms",
    "U": "us",
    "N": "ns",
    "M": "ME",
}


def _normalize_frequency(freq):
    """Normalize frequency aliases removed in pandas 3."""
    match = re.fullmatch(r"(\d*)([A-Za-z]+)(-.+)?", freq)
    if match is None:
        return freq

    multiplier, alias, suffix = match.groups()
    normalized_alias = _PANDAS_FREQUENCY_ALIASES.get(alias, alias)
    return f"{multiplier}{normalized_alias}{suffix or ''}"


def _get_frequency(index):
    """Get a pandas-compatible frequency string from an index."""
    freq = getattr(index, "freqstr", None)
    if freq is None and isinstance(index, pd.DatetimeIndex):
        try:
            freq = pd.infer_freq(index)
        except ValueError:
            freq = None

    return pd.tseries.frequencies.to_offset(_normalize_frequency(freq or "MS")).freqstr
