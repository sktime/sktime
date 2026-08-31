"""Compositors for splitters."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["SyncToLongest", "Repeat"]

from sktime.split.compose._sync_to_longest import SyncToLongest
from sktime.split.compose._repeat import Repeat
