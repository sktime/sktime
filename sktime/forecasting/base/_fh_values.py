# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Numpy-backed internal representation for ForecastingHorizonV2.

This module is pandas-free.
All data is stored in numpy arrays with associated metadata.
This is the "core" layer that the conversion layer
feeds into and that ForecastingHorizonV2 operates on.
"""

__all__ = []

from enum import Enum, auto

# core dependency
import numpy as np

# <check></check>
# this is the marker left to mark all delayed checks
# all occurences must be removed/addressed before merging this code


class FHValueType(Enum):
    """Enum describing the sematinc type of forecasting horizon values.

    <check>to be checked if any other attribute needs to be added.</check>

    Attributes
    ----------
    INT : integer steps (e.g. [1, 2, 3])
        Used for both relative integer horizons and absolute integer indices.
    TIMEDELTA : numpy timedelta64 durations (e.g. 1 day, 2 days)
        Used for relative time-based horizons.
    PERIOD : integer ordinals that represent pandas Period values
        Used for absolute period-based horizons. Requires freq.
    DATETIME : numpy datetime64 timestamps
        Used for absolute datetime-based horizons.
    """

    INT = auto()
    TIMEDELTA = auto()
    PERIOD = auto()
    DATETIME = auto()


# Which value types are inherently relative vs absolute
_RELATIVE_VALUE_TYPES = {FHValueType.INT, FHValueType.TIMEDELTA}
_ABSOLUTE_VALUE_TYPES = {FHValueType.INT, FHValueType.PERIOD, FHValueType.DATETIME}


class FHValues:
    """Lightweight, hashable, numpy-backed container for fh values + metadata.

    This class stores forecasting horizon values as sorted, deduplicated
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
        1-D numpy array of horizon values. Must be sorted and unique.
        dtype depends on ``value_type`` as below:
        INT: int64
        TIMEDELTA: timedelta64[ns]
        PERIOD: int64 (ordinal representation)
        DATETIME: datetime64[ns]
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
                f"instead got {type(values).__name__}"
            )
        if values.ndim != 1:
            raise ValueError(
                f"FHValues expects the value to be 1-D array, "
                f"instead got {values.ndim}-D array"
            )
        if not isinstance(value_type, FHValueType):
            raise TypeError(
                f"FHValues expects the value_type to be `FHValueType`, "
                f"instead got {type(value_type).__name__}"
            )
        if value_type == FHValueType.PERIOD and freq is None:
            raise ValueError(
                "freq can not be None when value_type is provided as PERIOD"
            )

        self.values = values
        self.value_type = value_type
        self.freq = freq
        self.timezone = timezone if value_type == FHValueType.DATETIME else None
        self.hash = None  # Cache for hash value, to be computerd lazily when needed

    @property
    def values(self) -> np.ndarray:
        """Return the underlying numpy array (read-only view)."""
        v = self.values.view()
        # Set the writeable flag to False to prevent mutation
        # to enforce quasi-immutability of the values array
        v.flags.writeable = False
        return v

    @property
    def value_type(self) -> FHValueType:
        """Return the semantic value type."""
        return self.value_type

    @property
    def freq(self) -> str | None:
        """Return frequency string or None."""
        return self.freq

    @property
    def timezone(self) -> str | None:
        """Return timezone string or None."""
        return self.timezone

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
        # its more of a placeholder right now.</check>
        if not isinstance(other, FHValues):
            return NotImplemented
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
        if self.hash is None:
            self.hash = hash(
                (
                    self.values.tobytes(),
                    # numpy arrays are not directly hashable, so we convert to bytes
                    self.value_type,
                    self.freq,
                    self.timezone,
                )
            )
        return self.hash

    def __repr__(self) -> str:
        # <check>to be implemented after checking the final set of
        # attributes to be stored</check>
        return (
            f"FHValues(values={self.values}, "
            f"value_type={self.value_type}, "
            f"freq={self.freq}, "
            f"timezone={self.timezone})"
        )

    def sort(self) -> "FHValues":
        """Return new FHValues with sorted values."""
        # <check>check the final set of attributes to be stored</check>
        # this will create a new instance with sorted values, same metadata
        sorted_vals = np.sort(self.values)
        return FHValues(sorted_vals, self.value_type, self.freq, self.timezone)

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

        For INT and PERIOD: checks consecutive integers.
        For TIMEDELTA and DATETIME: infers step from min diff, checks coverage.

        Returns
        -------
        bool
            True if values form a contiguous sequence.
        """
        if len(self.values) <= 1:
            return True

        sorted_vals = np.sort(self.values)

        if self.value_type in (FHValueType.INT, FHValueType.PERIOD):
            # <check>
            # multiple cases need to be hanfled here:
            #   - if the values are not perfectly regular
            #   - if there are duplicates
            #   - if there are missing values in between
            #   - and other cases
            # </check>
            expected_len = int(sorted_vals[-1] - sorted_vals[0]) + 1
            return len(self._values) == expected_len

        # <check>
        # same for TIMEDELTA or DATETIME
        # Below is partial implementation
        # </check>
        diffs = np.diff(sorted_vals)
        min_diff = diffs.min()
        if min_diff == np.timedelta64(0):
            return False  # duplicates
        total_span = sorted_vals[-1] - sorted_vals[0]
        expected_steps = total_span / min_diff
        return len(self.values) == int(expected_steps) + 1

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
        return result

    def copy(self) -> "FHValues":
        """Return a deep copy."""
        return FHValues(self.values.copy(), self.value_type, self.freq, self.timezone)
