# -*- coding: utf-8 -*-
"""Module exports: Series/Panel conversions."""

from sktime.datatypes._series_as_panel._convert import (
    convert_Series_to_Panel, convert_Panel_to_Series
)

__all__ = [
    "convert_Series_to_Panel",
    "convert_Panel_to_Series",
]
