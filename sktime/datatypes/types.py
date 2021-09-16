# -*- coding: utf-8 -*-
__all__ = ["Mtypes", "Scitypes"]

from typing import Union

from sktime.datatypes._registry import Scitype
from sktime.datatypes._panel import PanelMtype
from sktime.datatypes._series import SeriesMtype

# Types
Mtypes = Union[str, SeriesMtype, PanelMtype]
Scitypes = Union[str, Scitype]
