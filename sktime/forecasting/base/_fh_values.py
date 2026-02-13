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
        tz: str | None = None,
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
            raise ValueError("freq is required for PERIOD value type")

        self.values = values
        self.value_type = value_type
        self.freq = freq
        self.tz = tz
        self.hash = None  # Cache for hash value
