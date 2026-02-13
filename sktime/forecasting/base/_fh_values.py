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

# check-here
# this is the marker left to mark all delayed checks
# all occurences must be removed/addressed before merging this code


class FHValueType(Enum):
    """Enum describing the sematinc type of forecasting horizon values.

    check-here
    to be checked if any other attribute needs to be added.

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
