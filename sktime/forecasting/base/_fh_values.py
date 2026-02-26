# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Numpy-backed internal representation for ForecastingHorizon.

This module is pandas-free.
All data is stored in numpy arrays with associated metadata.
This is the "core" layer that the conversion layer
feeds into and that ForecastingHorizon operates on.
"""

__all__ = ["FHValueType", "FHValues", "VALID_FREQ_BASES", "validate_freq"]

import re
from enum import Enum, auto

# core dependency
import numpy as np

# sentinel for distinguishing "not provided" from None in _new()
_UNSET = object()

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


# <check></check>
# this is the marker left to mark all delayed checks
# all occurences must be removed/addressed before merging this code


class FHValueType(Enum):
    """Enum describing the sematinc type of forecasting horizon values.

    <check>to be checked if any other attribute needs to be added.</check>

    Attributes
    ----------
    INT : integer steps
        Used for both relative integer horizons and absolute integer indices.
        Stored as int64 values directly.
    TIMEDELTA : durations stored as int64 nanoseconds
        Used for relative time-based horizons.
    PERIOD : integer ordinals that represent pandas Period values
        Used for absolute period-based horizons. Requires freq.
    DATETIME : timestamps stored as int64 nanoseconds.
        Used for absolute datetime-based horizons.
    """

    INT = auto()
    TIMEDELTA = auto()
    PERIOD = auto()
    DATETIME = auto()


# Which value types can represent relative forecasting horizons
_RELATIVE_VALUE_TYPES = frozenset({FHValueType.INT, FHValueType.TIMEDELTA})

# Which value types can represent absolute forecasting horizons
_ABSOLUTE_VALUE_TYPES = frozenset(
    {FHValueType.INT, FHValueType.PERIOD, FHValueType.DATETIME}
)


class FHValues:
    """Lightweight, hashable, numpy-backed container for fh values + metadata.

    This class stores forecasting horizon values as sorted and deduplicated
    numpy arrays together wiath metadata (value type, frequency string).
    <check>
    at each step in implementation check if any other information
    needs to be stored as metadata for easy conversion to and from pandas.
    </check>
    This class is designed to make
    forecasting horizon internal representation pandas free.

    Note: Instances of this class are intended to be treated as quasi-immutable
    after creation. The ``values`` array should not be mutated externally.
    class methods should be used to create new instances rather than
    modifying existing ones as hashing will be implemented based on the
    content of values and metadata.
    <check>check for validation & enforecement of same</check>

    Parameters
    ----------
    values : np.ndarray
        1-D numpy array of horizon values with dtype int64.
        Will be sorted and deduplicated during construction.
        Semantic meaning depends on ``value_type``:
        INT: integer step counts
        TIMEDELTA: durations stored as int64 nanoseconds
        PERIOD: period ordinals
        DATETIME: timestamps stored as int64 nanoseconds
    value_type : FHValueType
        Semantic type of the stored values.
    freq : str or None, default=None
        Frequency string (e.g. "M", "D", "h").
        Required for PERIOD type; optional for DATETIME; ignored for others.
    timezone : str or None, default=None
        Timezone string for DATETIME values (e.g. "UTC", "US/Eastern").
        Ignored for non-DATETIME types.
    """

    def __init__(
        self,
        values: np.ndarray,
        value_type: FHValueType,
        freq: str | None = None,
        timezone: str | None = None,
    ):
        # validation of input types
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"FHValues expects the value to be passed as np.ndarray, "
                f"instead got {type(values)}"
            )
        if values.ndim != 1:
            raise ValueError(
                f"FHValues expects the value to be 1-D array, "
                f"instead got {values.ndim}-D array"
            )
        if not isinstance(value_type, FHValueType):
            raise TypeError(
                f"FHValues expects the value_type to be `FHValueType`, "
                f"instead got {type(value_type)}"
            )
        if value_type == FHValueType.PERIOD and freq is None:
            raise ValueError(
                "freq can not be None when value_type is provided as PERIOD"
            )

        # deduplication and sorting of values
        # <check>check if this is the right place to do this, or if it should
        # be done in the conversion layer before creating FHValues instance
        # Depends on if we are making FHValues class a base sequence class
        # and decoupling the FH specific need of sorting and deduplication from it
        # For now, keeping it here as this is the only use case.
        # </check>
        values = np.unique(values)  # np.unique both sorts and deduplicates
        if len(values) == 0:
            # if values is empty, we can not infer value_type, freq or timezone
            # but we still want to allow creation of an empty FHValues instance
            # for this we will set value_type to INT and freq and timezone to None
            # as these are the most basic types that can represent an empty horizon
            # <check>check if this handling is ok,
            # or if we want to stop creation of empty FHValues instances,
            # or if we want to allow passing of value_type and other metadata
            # even for empty values</check>
            value_type = FHValueType.INT
            freq = None
            timezone = None
        self._values = values
        self._value_type = value_type
        self._freq = freq
        self._timezone = timezone if value_type == FHValueType.DATETIME else None
        self._hash = None  # Cache for hash value

    @property
    def values(self) -> np.ndarray:
        """Return the underlying numpy array (read-only view)."""
        v = self._values.view()
        # Set the writeable flag to False to prevent mutation
        # to enforce quasi-immutability of the values array
        v.flags.writeable = False
        return v

    @property
    def value_type(self) -> FHValueType:
        """Return the semantic value type."""
        return self._value_type

    @property
    def freq(self) -> str | None:
        """Return frequency string or None."""
        return self._freq

    @freq.setter
    def freq(self, value: str | None) -> None:
        """Set frequency string."""
        self._freq = value

    @property
    def timezone(self) -> str | None:
        """Return timezone string or None."""
        return self._timezone

    def is_relative_type(self) -> bool:
        """Whether the value type is a relative type."""
        return self.value_type in _RELATIVE_VALUE_TYPES

    def is_absolute_type(self) -> bool:
        """Whether the value type is an absolute type."""
        return self.value_type in _ABSOLUTE_VALUE_TYPES

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other):
        # <check>below implementation needs to be checked,
        # its more of a placeholder right now.
        # Update1: Added the check and corresponding error for
        # equality check between un-matching instance types. </check>
        if not isinstance(other, FHValues):
            raise TypeError(
                "Comparison is only supported between FHValues instances, "
                f"got {type(other)}"
            )
        return (
            self.value_type == other.value_type
            and self.freq == other.freq
            and self.timezone == other.timezone
            and np.array_equal(self.values, other.values)
        )

    def __hash__(self) -> int:
        # <check>
        # For hashing, we need to consider all attributes that define the
        # identity of the instance.
        # 1. meta attributes (value_type, freq, timezone) are directly hashable,
        #    check if any other meta attributes are added later on
        #    that need to be included in hashing.
        # 2. numpy arrays are not, so we need to convert them to a hashable form
        #   Option 1: convert to bytes for hashing
        #   Option 2: convert to tuple (but can be expensive for large arrays)
        #   Option 3: use a hash of the array content (e.g. using hashlib)
        #       more complex but can be more efficient for large arrays
        #   Needs checking for the best apporach.
        # </check>
        if self._hash is None:
            self._hash = hash(
                (
                    self.values.tobytes(),
                    # numpy arrays are not directly hashable, so we convert to bytes
                    self.value_type,
                    self.freq,
                    self.timezone,
                )
            )
        return self._hash

    def __repr__(self) -> str:
        cls = type(self).__name__
        vtype = self.value_type.name
        n = len(self.values)
        if n <= 6:
            vals_str = str(self.values.tolist())
        else:
            head = self.values[:3].tolist()
            tail = self.values[-3:].tolist()
            vals_str = f"[{head[0]}, {head[1]}, {head[2]}, ..., "
            vals_str += f"{tail[0]}, {tail[1]}, {tail[2]}]"
        parts = [f"values={vals_str}", f"value_type={vtype}"]
        if self.freq is not None:
            parts.append(f"freq={self.freq!r}")
        if self._timezone is not None:
            parts.append(f"timezone={self.timezone!r}")
        return f"{cls}({', '.join(parts)})"

    # may need this function as a helper for checking
    # immutability of values array.
    def sort(self) -> "FHValues":
        """Return new FHValues with sorted values."""
        # <check>check the final set of attributes to be stored</check>
        # this will create a new instance with sorted values, same metadata
        sorted_vals = np.sort(self.values)
        return FHValues(sorted_vals, self.value_type, self.freq, self.timezone)

    # may need this function as a helper for checking
    # immutability of values array.
    def unique(self) -> "FHValues":
        """Return new FHValues with unique values (preserves sort order)."""
        unique_vals = np.unique(self.values)
        return FHValues(unique_vals, self.value_type, self.freq, self.timezone)

    def max(self):
        """Return maximum value."""
        return self.values.max() if len(self.values) > 0 else None

    def min(self):
        """Return minimum value."""
        return self.values.min() if len(self.values) > 0 else None

    def nunique(self) -> int:
        """Return number of unique values."""
        return len(np.unique(self.values))

    def is_contiguous(self) -> bool:
        """Check if values form a contiguous sequence.

        separate checking logic to be implemented for different value types.
        Currently
        - for INT and PERIOD: checks consecutive integers.
        - for TIMEDELTA and DATETIME: infers step from min diff, checks coverage.
        more to be added

        Returns
        -------
        bool
            True if values form a contiguous sequence.
        """
        # check added because currently FHValues instances
        # can be created with no values
        if len(self.values) <= 1:
            return True

        if self.value_type in (FHValueType.INT, FHValueType.PERIOD):
            # contiguous means every integer between min and max is present
            expected_len = int(self._values[-1] - self._values[0]) + 1
            # the above check is complete because self._values is sorted and unique
            return len(self.values) == expected_len

        # TIMEDELTA or DATETIME: check uniform spacing
        diffs = np.diff(self.values)
        if diffs.min() <= 0:
            # This shouldn't happen after unique+sort check in constructor,
            # but adding this as a guard against any mutations to the values array
            return False
        min_diff = diffs.min()
        # all diffs should equal the minimum diff for uniform spacing
        return bool(np.all(diffs == min_diff))

    def __getitem__(self, key):
        """Support integer and slice indexing.

        Returns a new FHValues for slices/arrays, or a scalar for int index.
        """
        # <check>
        # support for all valid slices and indexing needs to be implemented.
        # </check>

        result = self.values[key]
        if isinstance(result, np.ndarray):
            return FHValues(result, self.value_type, self.freq, self.timezone)
        return result  # scalar int64 value, returned as is

    def _new(
        self,
        values: np.ndarray | None = None,
        value_type: FHValueType | None = None,
        freq: str | None = _UNSET,
        timezone: str | None = _UNSET,
    ) -> "FHValues":
        """Create a new FHValues with selectively replaced attributes.

        Parameters
        ----------
        values : np.ndarray, optional
            New values array. If None, copies current values.
        value_type : FHValueType, optional
            New value type. If None, uses current value type.
        freq : str or None, optional
            New freq. If not provided, uses current freq.
        timezone : str or None, optional
            New timezone. If not provided, uses current timezone.

        Returns
        -------
        FHValues
            New instance with replaced attributes.
        """
        return FHValues(
            values=self._values.copy() if values is None else values,
            value_type=self._value_type if value_type is None else value_type,
            freq=self._freq if freq is _UNSET else freq,
            timezone=self._timezone if timezone is _UNSET else timezone,
        )

    def copy(self) -> "FHValues":
        """Return a deep copy."""
        return FHValues(self.values.copy(), self.value_type, self.freq, self.timezone)
