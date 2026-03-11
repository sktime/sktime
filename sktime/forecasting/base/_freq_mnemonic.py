# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Standard time series frequency mnemonics, validation, and alias resolution.

This module is pandas-free. It defines:

- ``VALID_FREQ_BASES``: our canonical frequency base strings (e.g. ``"M"``,
``"D"``, ``"h"``). These are time-domain concepts, chosen by us, and
never change regardless of what pandas calls them.
- ``_FREQ_GROUPS``: equivalence groups of alias strings for the same time
concept (e.g. ``{"M", "ME"}`` are both "monthly"). Used as seed list
for dynamic resolution in ``_fh_utils.py``.
- ``_parse_freq()``: splits freq strings into ``(multiplier, base, suffix)``.
- ``validate_freq()``: normalizes known aliases and validates against
``VALID_FREQ_BASES``.

Note: our canonical bases (e.g. ``"M"``, ``"Y"``) match what pandas Period-context
APIs expect (``.to_period``, ``PeriodIndex.from_ordinals``, ``.asfreq``).
Pandas Offset-context APIs (``to_offset``)
may prefer different strings (``"ME"``, ``"YE"``).
See the frequency resolution system in ``_fh_utils.py`` for the dynamic mapping.

Kept in a separate file to avoid circular imports because it may be used by both
``_fh_v2`` (ForecastingHorizon) and ``_fh_utils`` (PandasFHConverter).
"""

__all__ = [
    "VALID_FREQ_BASES",
    "validate_freq",
    "_FREQ_GROUPS",
    "_parse_freq",
    "_ALIAS_TO_CANONICAL_STATIC",
]

import re

# Standard time series frequency base mnemonics.
# These are our canonical names for time-domain concepts.
# They never change, regardless of what pandas calls them.
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

# Equivalence groups: sets of base strings referring to the same
# fundamental time frequency. Based on time CONCEPTS, not pandas versions.
#
# Used as seed list for dynamic freq resolution in _fh_utils.py.
# The dynamic resolver uses pandas offset TYPES (MonthEnd, Day, etc.)
# as the stable bridge, so new aliases are auto-discovered when
# their offset type matches a known canonical.
#
# Only needs a manual update when pandas introduces an entirely new
# alias string AND removes all existing aliases in the group (highlu-unlikely).
_FREQ_GROUPS = [
    {"Y", "YE", "A", "AE"},  # yearly
    {"Q", "QE"},  # quarterly
    {"M", "ME"},  # monthly
    {"SM", "SME"},  # semi-monthly
    {"BQ", "BQE"},  # business quarterly
    {"BY", "BYE"},  # business yearly
]

# Static alias -> canonical mapping built from _FREQ_GROUPS.
# For each group, the canonical is the base in VALID_FREQ_BASES (if any),
# otherwise the shortest base in the group.
_ALIAS_TO_CANONICAL_STATIC = {}
for _group in _FREQ_GROUPS:
    _canonical_bases = _group & VALID_FREQ_BASES
    _canonical = (
        min(_canonical_bases, key=len) if _canonical_bases else min(_group, key=len)
    )
    for _alias in _group:
        _ALIAS_TO_CANONICAL_STATIC[_alias] = _canonical

# Regex for validation: optional multiplier + canonical base + optional anchor
_FREQ_PATTERN = re.compile(
    r"^(\d+)?("
    + "|".join(sorted(VALID_FREQ_BASES, key=len, reverse=True))
    + r")(-.+)?$"
)

# General parsing regex: any base (not just canonical)
_PARSE_PATTERN = re.compile(r"^(\d+)?([A-Za-z]+)(-.+)?$")


def _parse_freq(freq_str):
    """Parse a frequency string into (multiplier, base, suffix).

    Parameters
    ----------
    freq_str : str or None
        Frequency string to parse. None returns ``("", "", "")``.

    Returns
    -------
    tuple of (str, str, str)
        ``(multiplier, base, suffix)``. Multiplier and suffix may be
        empty strings.

    Examples
    --------
    >>> _parse_freq("2D")
    ('2', 'D', '')
    >>> _parse_freq("M")
    ('', 'M', '')
    >>> _parse_freq("W-SUN")
    ('', 'W', '-SUN')
    >>> _parse_freq("2QE-DEC")
    ('2', 'QE', '-DEC')
    """
    if freq_str is None:
        return ("", "", "")
    m = _PARSE_PATTERN.match(freq_str)
    if m is None:
        return ("", freq_str, "")
    return (m.group(1) or "", m.group(2), m.group(3) or "")


def validate_freq(freq_str):
    """Validate a frequency string, normalizing pandas aliases first.

    Normalizes known pandas aliases (e.g. ``"ME"`` -> ``"M"``,
    ``"QE-DEC"`` -> ``"Q-DEC"``) using the static equivalence groups,
    then validates the base against ``VALID_FREQ_BASES``.

    Accepted format: optional integer multiplier + canonical base +
    optional anchor suffix (e.g. ``"M"``, ``"2D"``, ``"W-SUN"``).

    Parameters
    ----------
    freq_str : str
        Frequency string to validate.

    Returns
    -------
    str
        The normalized, validated frequency string.

    Raises
    ------
    ValueError
        If the normalized string doesn't match any accepted pattern.

    Examples
    --------
    >>> validate_freq("M")
    'M'
    >>> validate_freq("ME")
    'M'
    >>> validate_freq("2D")
    '2D'
    >>> validate_freq("W-SUN")
    'W-SUN'
    """
    mult, base, suffix = _parse_freq(freq_str)
    canonical_base = _ALIAS_TO_CANONICAL_STATIC.get(base, base)
    canonical = f"{mult}{canonical_base}{suffix}"
    if _FREQ_PATTERN.match(canonical):
        return canonical
    raise ValueError(
        f"Invalid frequency string: {freq_str!r}. "
        f"Expected an optional integer multiplier followed by one of "
        f"{sorted(VALID_FREQ_BASES)}, e.g. 'M', '2D', '4h', '15min'."
    )
