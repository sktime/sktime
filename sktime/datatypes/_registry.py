# -*- coding: utf-8 -*-

from sktime.datatypes._series._registry import (
    MTYPE_REGISTER_SERIES,
    MTYPE_LIST_SERIES,
)

from sktime.datatypes._panel._registry import (
    MTYPE_REGISTER_PANEL,
    MTYPE_LIST_PANEL,
)

MTYPE_REGISTER = MTYPE_REGISTER_SERIES + MTYPE_REGISTER_PANEL


__all__ = [
    "MTYPE_REGISTER",
    "MTYPE_LIST_PANEL",
    "MTYPE_LIST_SERIES",
    "SCITYPE_REGISTER"
]


SCITYPE_REGISTER = [
    ("Series", "uni- or multivariate time series"),
    ("Panel", "panel of uni- or multivariate time series"),
]
