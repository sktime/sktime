# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Standard time series frequency mnemonics and validation.

This module is pandas-free. It defines the accepted frequency base strings
(e.g. ``"M"``, ``"D"``, ``"h"``, etc) and provides validation for frequency strings
with optional integer multipliers (e.g. ``"2D"``, ``"15min"``).

Used by both ``_fh_v2`` (ForecastingHorizon) & ``_fh_utils`` (PandasFHConverter)
as a shared dependency to avoid circular imports.
"""

__all__ = ["VALID_FREQ_BASES", "validate_freq"]

import re

# Standard time series frequency base mnemonics.
# These are well-established, domain-standard frequencies used in
# time series forecasting. Pandas-specific variants (e.g. "BM", "MS", "QS")
# are handled by PandasFHConverter when extracting from pandas objects.
VALID_FREQ_BASES = frozenset(
    {
        "Y",  # yearly
        "Q",  # quarterly
        "M",  # monthly
        "W",  # weekly
        "D",  # daily
        "h",  # hourly
        "min",  # minutely
        "s",  # secondly
        "ms",  # millisecond
        "us",  # microsecond
        "ns",  # nanosecond
    }
)

# Regex: optional integer multiplier followed by a frequency base
_FREQ_PATTERN = re.compile(
    r"^(\d+)?(" + "|".join(sorted(VALID_FREQ_BASES, key=len, reverse=True)) + r")$"
)


def validate_freq(freq_str):
    """Validate a frequency string against accepted standard values.

    Accepted format is an optional integer multiplier followed by a base
    frequency mnemonic, e.g. ``"M"``, ``"2D"``, ``"4h"``, ``"15min"``.

    Parameters
    ----------
    freq_str : str
        Frequency string to validate.

    Returns
    -------
    str
        The validated frequency string (unchanged).

    Raises
    ------
    ValueError
        If ``freq_str`` does not match any accepted frequency pattern.

    Examples
    --------
    >>> validate_freq("M")
    'M'
    >>> validate_freq("2D")
    '2D'
    >>> validate_freq("15min")
    '15min'
    """
    if _FREQ_PATTERN.match(freq_str):
        return freq_str
    raise ValueError(
        f"Invalid frequency string: {freq_str!r}. "
        f"Expected an optional integer multiplier followed by one of "
        f"{sorted(VALID_FREQ_BASES)}, e.g. 'M', '2D', '4h', '15min'."
    )
