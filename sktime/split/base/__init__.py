# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements base classes for splitting in sktime."""

__all__ = [
    "BaseSplitter",
    "BaseWindowSplitter",
]

from sktime.split.base._base_splitter import BaseSplitter
from sktime.split.base._base_windowsplitter import BaseWindowSplitter
