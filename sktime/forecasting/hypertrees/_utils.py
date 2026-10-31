"""Utilities for Hyper-Trees forecasters."""

import re

import pandas as pd
from packaging.version import Version

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
    """Get a pandas-compatible frequency string from an index.

    Normalize legacy aliases only when running with pandas >= 3.0.0. Older
    pandas versions accept the legacy aliases (or map them themselves), so
    avoid forcing the newer canonical aliases which might not exist there.
    """
    freq = getattr(index, "freqstr", None)
    if freq is None and isinstance(index, pd.DatetimeIndex):
        try:
            freq = pd.infer_freq(index)
        except ValueError:
            freq = None

    freq = freq or "MS"

    # Only normalize legacy aliases for pandas 3+, where old aliases may be
    # rejected. For older pandas, keep the original and let pandas accept or
    # map it as appropriate for that runtime.
    if Version(pd.__version__) >= Version("3.0.0"):
        freq = _normalize_frequency(freq)

    return pd.tseries.frequencies.to_offset(freq).freqstr
