# -*- coding: utf-8 -*-
"""Defines types for the datatypes module."""
__all__ = ["Mtype", "SciType"]

from typing import Union

from sktime.datatypes._panel import PanelMtype
from sktime.datatypes._series import SeriesMtype
from sktime.datatypes._datatypes import Datatypes

# Types
Mtype = Union[str, SeriesMtype, PanelMtype]
SciType = Datatypes
